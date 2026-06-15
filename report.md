# Praktikum 2 — Report

**Distributed Job Operator: declarative, self-optimizing distributed ML training on Kubernetes**

---

## 1. Abstract

This praktikum delivers a Kubernetes operator (`distributed-training-operator`) that automates
the full lifecycle of a distributed machine-learning training job on GKE — provisioning an
ephemeral node pool, submitting the training workload, monitoring it, collecting metrics and
cost, and tearing the infrastructure back down — from a single declarative `kubectl apply`.

Beyond automation, the operator embeds a **topology solver**: instead of hard-coding a node
count, the user states an objective (`targetTime`, `maxCost`) and the operator fits a scaling
model from its own historical runs and selects the node count that meets the Objective. The solver
"learns" from every run it executes.

The operator was validated against the manual workflow from Praktikum 1 across **8 measured
runs** (2 baseline, 2 calibration, 4 objective-driven). The headline results:

- **Human effort dropped from 5 commands per run to 1** (`kubectl apply`).
- **No training overhead** — loss, perplexity, throughput and cost are statistically
  identical between the manual baseline and the operator at the same node count.
- **The solver's training-time prediction converged to 0.8 % error** in steady state
  (≤ 6.7 % even on its very first objective-driven invocation), and adapted its node-count
  decision to the Objective (6 → 7 → 7 → 11 nodes across changing objectives).

---

## 2. Motivation and Problem Statement

Running distributed ML training on cloud infrastructure has two recurring pain points:

1. **Right-sizing.** Too few nodes miss the deadline; too many waste money on the diminishing
   returns of parallelism. The "correct" node count is workload-specific and not knowable
   a priori.
2. **Infrastructure lifecycle.** Spinning up a node pool, applying the job, watching it,
   copying results out, and deleting the pool is a repetitive, error-prone, manual sequence
   with no retry semantics, no cost tracking, and no memory across runs.

The Praktikum 1 workflow exhibited exactly this: a worker pool provisioned by `gcloud`, a
`PyTorchJob` applied by hand, metrics pulled with `kubectl cp`, and teardown done manually.
The guiding question for Praktikum 2 was: **can this be replaced by one declarative command
that also decides the cluster size for you?**

---

## 3. The Operator

### 3.1 Custom Resources

The operator is built with Kubebuilder / controller-runtime and introduces two CRDs in the
API group `training.distributedtraining.io/v1`:

| Kind | Role |
|---|---|
| `DistributedTraining` | The user-facing resource. Declares the model, dataset, training hyperparameters, hardware, and either an explicit `topology.nodes` **or** an `objective`. |
| `DistributedTrainingHistory` | An operator-written record of a completed run (nodes, throughput, total work, provisioning time, cost, machine type, config hash). The training data for the solver. |

### 3.2 Job Lifecycle (reconcile state machine)

Each `DistributedTraining` advances through a phase state machine driven by the controller:

```
Pending → Provisioning → Ready → Running → Collecting → Succeeded
                                                       ↘ Failed
```

| Phase | What the operator does |
|---|---|
| **Pending** | Validates the spec; if `spec.objective` is set, runs the topology solver and writes `status.resolvedTopology`. |
| **Provisioning** | Creates an ephemeral GKE node pool; polls until the requested number of nodes are `Ready`. |
| **Ready** | Submits the backend job manifest (PyTorchJob / SparkApplication / Job). |
| **Running** | Monitors backend job conditions and pod logs. |
| **Collecting** | Scrapes metrics (loss, throughput, cost), writes a `DistributedTrainingHistory` entry, and deletes the node pool (asynchronously, in parallel with result recording). |
| **Succeeded** | Records final results in `status.results`; node pool already gone. |
| **Failed** | Terminal; node pool deleted; error written to `status.message`. |

### 3.3 Pluggable backends

The controller has no workload-specific imports. Backends implement a `JobBackend` interface
(manifest generation, phase polling, start/end-time extraction, result collection):

| Backend | Resource created | Use case |
|---|---|---|
| `pytorch` (default) | Kubeflow `PyTorchJob` | LLM fine-tuning, distributed deep learning |

Training-time measurement is deliberately defined as the PyTorchJob
`status.startTime → status.completionTime` window, so `training_seconds` excludes pod
scheduling and image-pull time and is directly comparable to the manual baseline.

### 3.4 Pluggable cloud providers

Node-pool lifecycle is abstracted behind a `cloud.Provider` interface
(`CreateNodePool` / `DeleteNodePool` / `IsOperationDone` / `NodePoolLabelKey`). GKE is
implemented; the interface documents the EKS/AKS equivalents (label keys, machine types),
and a Kustomize overlay pattern (`config/overlays/<provider>/`) lets a new provider be wired
in without touching the controller. All mutating operations are non-blocking and polled by
operation ID, matching the reconcile model.

### 3.5 History tracking

After every successful run the operator writes a `DistributedTrainingHistory` CR. Runs of "the
same job" are grouped by a **SHA-256 config hash** of the workload-defining fields (backend,
machine type, model, dataset, batch size, epochs, etc.), so the solver only ever learns from
comparable runs. At most **10** entries are retained per config hash (oldest evicted first).

### 3.6 Topology solver

When `spec.objective` is set, the user declares constraints instead of a node count:

```yaml
spec:
  objective:
    targetTime: "10m"   # max acceptable wall-clock training time
    maxCost: "10"       # max spend in USD
    maxNodes: 10        # hard upper bound
```

The solver fits a **scaling-overhead model** to recorded throughput:

```
η(n) = 1 / (1 + α(n − 1))            # parallel efficiency
P(n) = p_baseline × n × η(n)          # predicted throughput at n nodes
T(n) = W / P(n)                       # predicted training time
Cost(n) = C_h × n × (T(n) + T_prov_avg) / 3600
```

| Symbol | Meaning |
|---|---|
| `α` | serial fraction — how fast efficiency decays as nodes are added (captures gloo DDP collective-comms cost) |
| `p_baseline` | throughput per node at the smallest observed node count |
| `W` | total work (samples to process), constant per workload |
| `T_prov_avg` | average provisioning time across history |
| `C_h` | hourly per-node cost, read from the `machine-costs` ConfigMap |

**Fitting α** depends on how much history exists:

- 1 distinct node count → `α = 0` (linear scaling assumed).
- **2 distinct node counts → closed-form fit** (one equation, one unknown).
- 3+ distinct node counts → 1-D golden-section search minimizing squared residuals.

**Calibration.** With no history for a config hash, the solver cannot estimate the model, so
it schedules a **calibration run** at `--default-calibration-nodes` (default 2). The
objective is applied from the next run onward, once real throughput data exists.

**Constraint resolution.** `targetTime` → smallest `n` with `T(n) ≤ targetTime`; `maxCost` →
largest affordable `n`; both → satisfy time first, fall back to the largest affordable `n`
with a warning if cost is violated.

### 3.7 Observability

The operator exports Prometheus metrics (nodes provisioned, provisioning/training seconds,
actual cost, predicted vs. actual time/cost, fitted α, solver-selected nodes, training
quality) scraped via a `ServiceMonitor`. A pre-built Grafana dashboard
(`config/grafana/distributedtraining-dashboard.json`) visualizes them. Baseline (manual) runs push
the same metric set to a Prometheus Pushgateway so both approaches land on one dashboard.

---

## 4. Infrastructure and Deployment

The full stack is provisioned with Terraform (`main.tf`, `network.tf`, `nfs.tf`,
`grafana.tf`):

- VPC + subnet and a GKE cluster (`praktikum2`, zone `us-east1-d`) with a general-purpose
  node pool.
- `gke-cluster-sa` IAM service account, role bindings, and Workload Identity bindings.
- NFS server backed by a GCE persistent disk, exposing a shared `distributed-training-output-pvc`
  that every training pod mounts at `/mnt/output` (also hosts the shared HuggingFace cache
  at `/exports/hf-cache`, persistent across node-pool teardowns).
- `kube-prometheus-stack` (Prometheus + Grafana) + Prometheus Pushgateway in namespace
  `monitoring`, plus the DistributedTraining dashboard ConfigMap.

The operator deploys via a Kustomize GKE overlay (`config/overlays/gke/`). The Kubeflow
training-operator (PyTorchJob CRD) is installed separately. The development workflow,
including the KodeKloud GCP playground constraints (org policies on disk size, IAM, IPv6) and
their workarounds, is documented in `operator/README.md` and `RESTORATION.md`.

---

## 5. Experiments

### 5.1 Design

The operator was evaluated against the manual Praktikum 1 workflow. Both approaches run on
identical hardware and identical training arguments, so training **quality** is expected to
match — the comparison is about **workflow**: human effort, provisioning/training time, cost,
and (for the operator only) **solver prediction accuracy and adaptation**.

**Fixed workload across all runs:**

| Parameter | Value |
|---|---|
| Model | `BEE-spoke-data/smol_llama-101M-GQA` |
| Dataset | `medalpaca/medical_meadow_medical_flashcards` (train) |
| Machine type | `c4-highcpu-8` — **$0.42/hr per node** |
| Training | 1 epoch, batch 4, LR 2e-5, 20 % validation split |
| Adapters | LoRA (rank 4, α 8, dropout 0.1), bf16 mixed precision |
| DDP backend | gloo (CPU) |

> The config hash is a SHA-256 of these fields — keeping them constant is what lets the
> solver treat all runs as the same workload.

### 5.2 Run matrix (8 runs)

| Run ID | Approach | Nodes | Purpose |
|---|---|---|---|
| B-2 | Baseline (manual) | 2 | Manual 2-node distributed |
| B-3 | Baseline (manual) | 3 | Manual 3-node distributed |
| O-CAL-2 | Operator | 2 (fixed) | Calibration — writes 2-node history |
| O-CAL-3 | Operator | 3 (fixed) | Calibration — writes 3-node history |
| O-OBJ-A1 | Operator | solver | Objective A first solver run |
| O-OBJ-A2 | Operator | solver | Objective A repeat |
| O-OBJ-A3 | Operator | solver | Objective A repeat |
| O-OBJ-max | Operator | solver | Different objective (agressive time constraint) |

Order: B-2 → B-3 → O-CAL-2 → O-CAL-3 → O-OBJ-A1 → A2 → A3 → max. The objective runs must
follow the calibration runs (they need ≥ 2 history entries to fit α), and A1→A2→A3 is ordered
because each reads the history written by its predecessors.

### 5.3 Results

#### Table 1 — Workflow comparison at matched node count

| Run | Approach | Nodes | Human steps | Provisioning (s) | Training (s)¹ | Cost (USD)² | Eval loss | Perplexity | Samples/s |
|---|---|---|---|---|---|---|---|---|---|
| B-2 | Baseline | 2 | **5** | 65 | 1444 | 0.352 | 1.5512 | 4.717 | 21.87 |
| O-CAL-2 | Operator | 2 | **1** | ~83 | 1237³ | 0.353 | 1.5512 | 4.717 | 21.97 |
| B-3 | Baseline | 3 | **5** | 69 | 1018 | 0.380 | 1.5603 | 4.760 | 31.55 |
| O-CAL-3 | Operator | 3 | **1** | ~82 | 879³ | 0.380 | 1.5603 | 4.760 | 30.90 |

¹ PyTorchJob `startTime → completionTime` (excludes image pull/scheduling, symmetric for both).

² `cost = nodes × 0.42 × (prov + train) / 3600`.

³ Operator calibration `train_runtime` (inner training loop); wall-clock is comparable to baseline.

**Reading of Table 1:** at the same node count the operator produces **bit-identical eval loss
and perplexity** and statistically identical throughput — confirming it does not interfere with
training — while collapsing the per-run human effort from **5 commands to 1**. Provisioning and
cost are comparable; the operator adds no measurable overhead.

![Pod runtime — baseline vs operator at the same N](results/grafana-results/pod-runtime-comparison-slide3.png)

#### Table 2 — Solver accuracy and adaptation (objective-driven runs)

| Run | Objective | Solver nodes | Pred. time | Actual time | **Time err.** | Pred. cost | Actual cost | Cost err. | α at solve |
|---|---|---|---|---|---|---|---|---|---|
| O-OBJ-A1 | A: ≤10 m, ≤$10, ≤10n | 6 | 9 m 28 s | 10 m 06 s | **6.7 %** | $0.379 | $0.402 | 5.6 % | 0.0330 |
| O-OBJ-A2 | A (same) | 7 | 8 m 56 s | 9 m 25 s | **5.4 %** | $0.421 | $0.440 | 4.3 % | 0.0473 |
| O-OBJ-A3 | A (same) | 7 | 9 m 13 s | 9 m 09 s | **0.8 %** | $0.433 | $0.429 | 0.8 % | 0.0541 |
| O-OBJ-max | B: ≤7 m, ≤$10, ≤15n | 11 | 6 m 52 s | 6 m 39 s | **3.1 %** | — | — | — | 0.0549 |

**Solver learns from data.** The fitted scaling-overhead α rises and stabilizes as history
accumulates (0.033 → 0.047 → 0.054 → 0.055), and the training-time prediction error drops to
**0.8 %** by the third objective run — the model is fit from just two calibration runs and
predicts every subsequent run within single-digit percent. Cost prediction tracks similarly
(≤ 5.6 %, converging to 0.8 %).

**Solver adapts to the Objective.** Given the **same workload and the same fitted α** but a tighter
time budget (Objective B: ≤ 7 min), the solver selects **11 nodes** instead of the 6–7 it chose
for Objective A. The manual baseline cannot express this decision at all — `--nnodes=N` is a
hand-picked constant with no notion of an objective.

![Solver-selected nodes adapt to the objective](results/grafana-results/grafana-slide5-adaptation.png)

![Scaling-overhead α converges across runs](results/grafana-results/grafana-slide5-alpha-convergence-values.png)

![Training time: predicted vs actual, and prediction error %](results/grafana-results/grafana-slide5-solver-accuracy.png)

---

## 6. Findings

1. **Automation with zero quality cost.** The operator reproduces the baseline's training
   outcome exactly (identical loss/perplexity at matched N) while reducing the workflow to a
   single declarative command and tracking cost automatically.
2. **The scaling model is accurate and improves with use.** Two calibration runs are enough to
   predict training time within ~7 %; by the third objective run the error is sub-1 %. The
   fitted α stabilizes as history grows, the empirical signature of "the solver gets smarter as
   you use it."
3. **Objective-driven scaling is a genuine new capability.** The operator turns an open-ended
   right-sizing problem into a declarative constraint and solves it from real measured data —
   something the manual workflow has no mechanism to do.

---

## 7. Deliverables

| Deliverable | Location |
|---|---|
| Operator source (Go, kubebuilder) | `operator/` — `api/v1/`, `internal/controller/`, `internal/cloud/gke/`, `internal/backend/{pytorch,spark,collect}/` |
| CRDs | `operator/config/crd/bases/training.distributedtraining.io_*.yaml` |
| Topology solver | `operator/internal/controller/solver.go` (+ `TestRunSolver.md`) |
| Deployment (Kustomize + GKE overlay) | `operator/config/`, `operator/config/overlays/gke/` |
| Container image | `jmukaj/distributed-training-operator` |
| Unit + e2e tests, CI | `operator/internal/controller/*_test.go`, `operator/test/e2e/`, `operator/.github/workflows/` |
| Operator documentation | `operator/README.md` |
| Training image + fine-tune workload (artifact from P1) | `fine-tune/` (CPU/GPU data-parallel `finetune.py`, Dockerfiles, Helm data-chart) |
| Infrastructure as code | `main.tf`, `network.tf`, `nfs.tf`, `grafana.tf`, `*.tfvars` |
| Monitoring | Grafana dashboard `operator/config/grafana/distributedtraining-dashboard.json`, Pushgateway baseline scripts `scripts/` |
| Experiment plan | `experiments.md` |
| Raw experiment data | `results/` (`*-timings.txt`, `*-metrics.txt`, `*-all_results.json`, `*-status.yaml`, pool logs) |
| Grafana screenshots | `results/grafana-results/`, `grafana-distributed-training/` |
| Proposal | `p2-proposal-revised.docx` |

---

## 8. Limitations and Future Work

- **Constant-α assumption.** A single serial fraction does not capture non-linear scaling
  overhead at higher node counts. A per-N or piecewise model would extend the accurate range.
- **Evaluation time folded into α.** HuggingFace post-training evaluation is fixed overhead
  but is currently absorbed into the scaling fit; modeling a separate `T_eval` term would
  sharpen predictions.
- **Single cloud validated.** The provider abstraction is in place for EKS/AKS but only GKE is
  implemented and tested.
