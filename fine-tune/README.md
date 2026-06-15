kubectl apply -f nfs.yaml
helm install --namespace kubeflow-user-example-com -f data-chart/cpu-medical_meadow_values.yaml llama2-distributed ./data-chart
helm install --namespace kubeflow-user-example-com -f data-chart/gpu-medical_meadow_values.yaml llama2-distributed ./data-chart

helm uninstall --namespace kubeflow-user-example-com  llama2-distributed

kubectl cp --retries 10 --namespace kubeflow-user-example-com medical-meadow-dataaccess:/tmp/output/ .

BEE-spoke-data/smol_llama-101M-GQA  RAM used 4GB




Google Compute Engine: Not all instances running in IGM after 35m11.182245706s. Expected 1, running 0, transitioning 1. Current errors: [GCE_STOCKOUT]: Instance 'gke-cluster-2-pool-2-d1f7daf8-zb8d' creation failed: The zone 'projects/praktikum-475820/zones/us-central1-a' does not have enough resources available to fulfill the request. Try a different zone, or try again later.