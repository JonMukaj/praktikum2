# Praktikum 2 — Distributed Training Operator

A self-contained repository for **Praktikum 2**: a Kubernetes operator that automates
the full lifecycle of distributed ML training jobs on GKE, together with the Terraform
infrastructure to provision the cluster and the experiment plan used to evaluate it
against the manual workflow from Praktikum 1.

From a single `kubectl apply` of a `DistributedTraining` custom resource the operator:

1. **Solves** for an appropriate node count given an objective (`targetTime`, `maxCost`),
   using a scaling model fit from its own historical runs.
2. **Provisions** an ephemeral GKE node pool.
3. **Submits** the training workload (Kubeflow `PyTorchJob`, Spark `SparkApplication`,
   or a plain `batch/v1 Job`).
4. **Monitors** progress and collects metrics (loss, throughput, cost).
5. **Tears down** the node pool when training completes.
6. **Records** a `DistributedTrainingHistory` CR so future runs get smarter.

---

## Repository Layout

| Path | Contents |
| --- | --- |
| [operator/](operator/) | Operator source code (Go, kubebuilder), CRDs, controllers, topology solver, Dockerfile, Makefile |
| [main.tf](main.tf), [network.tf](network.tf), [nfs.tf](nfs.tf), [grafana.tf](grafana.tf), [provider.tf](provider.tf), [variables.tf](variables.tf) | Terraform to provision the GKE cluster, VPC, NFS, and Grafana |
| [kubeflow.tfvars](kubeflow.tfvars), [cpu-small-kubeflow.tfvars](cpu-small-kubeflow.tfvars), [cpu-large-kubeflow.tfvars](cpu-large-kubeflow.tfvars), [gpu-kubeflow.tfvars](gpu-kubeflow.tfvars), [operator-cpu-small-kubeflow.tfvars](operator-cpu-small-kubeflow.tfvars) | Per-scenario Terraform variable files |
| [fine-tune/](fine-tune/) | Training workload (Helm chart, PyTorchJob manifests, NFS data jobs, scripts) |
| [scripts/](scripts/) | Baseline (manual workflow) automation scripts used in Praktikum 1 comparison |
| [results/](results/) | Raw measurement output for all 8 experiment runs (timings, metrics, applied manifests, CR statuses) |
| [images/](images/) | Plots used in the report (training time, cost, scaling) |
| [presentation/](presentation/) | Final presentation slides |
| [experiments.md](experiments.md) | **Experiment plan** — how to reproduce every run, baseline and operator |
| [report.md](report.md) | **Final report** — abstract, methodology, results, and discussion |

---

## Quick Start

Prerequisites: `gcloud`, `terraform`, `kubectl`, `kustomize`, a GCP project with billing
enabled, and quotas for the chosen machine type.

### 1. Provision the cluster

```bash
gcloud auth login
terraform init
terraform plan  -var-file=kubeflow.tfvars
terraform apply -var-file=kubeflow.tfvars
gcloud container clusters get-credentials praktikum1 \
    --zone us-east1-d --project <your-project-id>
```

Use a different `*.tfvars` file to switch profiles (small CPU / large CPU / GPU /
operator-only). See [variables.tf](variables.tf) for the full schema.

### 2. Install Kubeflow

Install the standalone Kubeflow training-operator (CRDs + controller in the `kubeflow`
namespace — no other Kubeflow components required):

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"

# Verify
kubectl get crd pytorchjobs.kubeflow.org
kubectl wait --for=condition=Available deploy/training-operator -n kubeflow --timeout=2m
```

### 3. Deploy the NFS server and shared PVC

[nfs.yaml](nfs.yaml) runs the NFS server backed by the GCE disk created by Terraform.
[operator/config/samples/pvc.yaml](operator/config/samples/pvc.yaml) creates the
`PersistentVolume` and `distributed-training-output-pvc` PVC that every training pod
mounts at `/mnt/output`. Apply in order:

```bash
kubectl apply -f nfs.yaml
kubectl wait --for=condition=Ready pod -l role=nfs-server --timeout=2m

kubectl apply -f operator/config/samples/pvc.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/distributed-training-output-pvc --timeout=1m
```

### 4. Deploy the operator

```bash
cd operator
make install         # CRDs
make deploy IMG=<your-registry>/distributed-training-operator:latest
```

See [operator/README.md](operator/README.md) for the full controller documentation,
CRD reference, and topology-solver details.

### 5. Run a training job

The example below matches the calibration run used in the experiments (2-node CPU,
`smol_llama-101M-GQA` fine-tuned on `medical_meadow_medical_flashcards`).
See [experiments.md](experiments.md) for the full run matrix.

```yaml
apiVersion: training.praktikum.io/v1alpha1
kind: DistributedTraining
metadata:
  name: distributedtraining-cal-2node
  namespace: default
spec:
  backend: pytorch
  hardware:
    type: cpu
    machineType: "c4-highcpu-8"
  topology:
    nodes: 2
    processesPerNode: 1
  outputPVCName: distributed-training-output-pvc
  model:
    name: BEE-spoke-data/smol_llama-101M-GQA
    trainingScript: /workspace/scripts/finetune.py
  dataset:
    name: medalpaca/medical_meadow_medical_flashcards
    split: train
  training:
    batchSize: 4
    learningRate: "2e-5"
    epochs: 1
    useFastTokenizer: false
  pytorchSpec:
    image: jmukaj/cpu-fine-tuner-hf:v1.0.0
    resources:
      requests: {cpu: "4", memory: "10Gi"}
      limits:   {cpu: "4", memory: "10Gi"}
```

```bash
kubectl apply -f results/O-CAL-2-distributedtraining.yaml

# Watch phase transitions: Pending → Provisioning → Ready → Running → Collecting → Succeeded
kubectl get dj distributedtraining-cal-2node -w

# Operator logs
kubectl logs -f -n distributed-training-system deploy/distributed-training-controller-manager
```

The operator provisions the node pool, submits the PyTorchJob, collects results into
`status.results`, and deletes the node pool when done.

### 6. Tear down

```bash
terraform destroy -var-file=kubeflow.tfvars -refresh=false
```

---

## Experiments

The full experimental methodology — workload definition, run matrix (2 baseline,
2 calibration, 4 objective-driven), step-by-step reproduction commands, and metric
collection procedure — is in [experiments.md](experiments.md).

Raw outputs for every run are in [results/](results/) and are referenced directly
by the report.

---

## Report

The written deliverable — abstract, design of the operator and topology solver,
comparison against the manual Praktikum 1 baseline, plots, discussion, and
conclusions — is in [report.md](report.md).

---

## License

See [operator/](operator/) for the operator's license. The rest of this repository
is provided for academic purposes as part of the Praktikum 2 deliverable.
