#!/usr/bin/env bash
# baseline_pool.sh — Provision or tear down the baseline worker node pool via
# gcloud, measuring wall-clock duration end-to-end.
#
# Mirrors the (now-commented) google_container_node_pool.worker resource in
# main.tf so baseline runs use the same node config the operator would.
#
# Usage:
#   ./scripts/baseline_pool.sh provision <run_id> <num_nodes>
#   ./scripts/baseline_pool.sh teardown  <run_id>
#
# Examples:
#   ./scripts/baseline_pool.sh provision B-2 2
#   ./scripts/baseline_pool.sh teardown  B-2
#
# Outputs:
#   results/<run_id>-pool-provision.log     gcloud + kubectl output
#   results/<run_id>-pool-teardown.log      gcloud output
#   results/<run_id>-timings.txt            appended with t_*/provisioning_s/teardown_s

set -euo pipefail

# ── cluster config (matches operator-cpu-small-kubeflow.tfvars) ───────────────
PROJECT=praktikum2-494215
CLUSTER=praktikum2
ZONE=us-east1-d
POOL_NAME=${CLUSTER}-worker-node-pool
SERVICE_ACCOUNT=gke-cluster-sa@${PROJECT}.iam.gserviceaccount.com

# ── arg parsing ───────────────────────────────────────────────────────────────
CMD=${1:-}
RUN_ID=${2:-}
NODES=${3:-}

if [[ -z "$CMD" || -z "$RUN_ID" ]]; then
  echo "Usage: $0 provision <run_id> <num_nodes>"
  echo "       $0 teardown  <run_id>"
  exit 1
fi

RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"
TIMINGS="$RESULTS_DIR/${RUN_ID}-timings.txt"

case "$CMD" in
  provision)
    if [[ -z "$NODES" ]]; then
      echo "provision requires <num_nodes>"; exit 1
    fi

    LOG="$RESULTS_DIR/${RUN_ID}-pool-provision.log"
    T0=$(date +%s)
    {
      echo "t_provision_start = $(date +%T)"
    } | tee -a "$TIMINGS"

    gcloud container node-pools create "$POOL_NAME" \
      --project="$PROJECT" \
      --cluster="$CLUSTER" \
      --zone="$ZONE" \
      --num-nodes="$NODES" \
      --machine-type=c4-highcpu-8 \
      --disk-size=50 \
      --scopes=cloud-platform \
      --service-account="$SERVICE_ACCOUNT" \
      --workload-metadata=GKE_METADATA \
      --node-labels=environment=production,role=worker \
      --tags=gke-node,"${PROJECT}-gke" \
      --node-taints=reserved-pool=true:NoSchedule \
      2>&1 | tee "$LOG"

    # gcloud returns when the operation is DONE, but the node may take a few
    # more seconds to register Ready with the kube apiserver.
    echo "Waiting for $NODES nodes in pool $POOL_NAME to become Ready..." | tee -a "$LOG"
    until [ "$(kubectl get nodes -l cloud.google.com/gke-nodepool="$POOL_NAME" \
        --no-headers 2>/dev/null | grep -c ' Ready ')" -ge "$NODES" ]; do
      sleep 2
    done

    T1=$(date +%s)
    {
      echo "t_provision_end   = $(date +%T)"
      echo "provisioning_s    = $((T1 - T0))"
    } | tee -a "$TIMINGS"
    ;;

  teardown)
    LOG="$RESULTS_DIR/${RUN_ID}-pool-teardown.log"

    # Defensive: ensure the file ends with a newline before appending so a
    # previous writer's missing \n doesn't concatenate with our first line.
    if [[ -s "$TIMINGS" && -n "$(tail -c1 "$TIMINGS")" ]]; then
      echo "" >> "$TIMINGS"
    fi

    T0=$(date +%s)
    {
      echo "t_teardown_start  = $(date +%T)"
    } | tee -a "$TIMINGS"

    gcloud container node-pools delete "$POOL_NAME" \
      --project="$PROJECT" \
      --cluster="$CLUSTER" \
      --zone="$ZONE" \
      --quiet \
      2>&1 | tee "$LOG"

    T1=$(date +%s)
    {
      echo "t_teardown_end    = $(date +%T)"
      echo "teardown_s        = $((T1 - T0))"
    } | tee -a "$TIMINGS"
    ;;

  *)
    echo "Unknown command: $CMD"
    echo "Usage: $0 provision <run_id> <num_nodes>"
    echo "       $0 teardown  <run_id>"
    exit 1
    ;;
esac
