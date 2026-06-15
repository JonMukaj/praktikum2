#!/usr/bin/env bash
# baseline_train.sh — Apply a baseline PyTorchJob, wait for completion,
# collect results, and write training/collection timings.
#
# training_s is the PyTorchJob's pod runtime — wall-clock from
# status.startTime (master pod reached Running) to status.completionTime
# (all replicas Succeeded). This matches the operator's
# distributedtraining_training_seconds, which records pod runtime only.
#
# apply_to_succeeded_s is the older "kubectl apply → PyTorchJob Succeeded"
# wall-clock kept as a secondary record: includes pod scheduling and image
# pull, which fall outside the operator's training metric.
#
# train_runtime / eval_runtime from all_results.json are recorded as
# inner-loop sanity-check values.
#
# Usage:
#   ./scripts/baseline_train.sh <run_id>
#
# Prerequisite: results/<run_id>-pytorchjob.yaml must already exist
#
# Outputs:
#   results/<run_id>-all_results.json    full HuggingFace results
#   results/<run_id>-metrics.txt         loss / perplexity / sps / runtimes
#   results/<run_id>-timings.txt         appended with t_*, training_s, collection_s
#   stdout                               ready-to-paste push_baseline.sh command

set -euo pipefail

RUN_ID=${1:-}
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <run_id>   (e.g. B-2)"
  exit 1
fi

# K8s namespace where the PyTorchJob, NFS server, and PVC live — used for
# all kubectl queries.
K8S_NAMESPACE=default
# Label value pushed to Pushgateway. Matches the operator's metric labeling
# (Prometheus's scrape relabel rewrites the operator's `namespace` label to
# its own pod namespace, so baseline metrics must use the same value for the
# Grafana dropdown to show baseline and operator side-by-side).
METRIC_NAMESPACE=default
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
TIMINGS="$RESULTS_DIR/${RUN_ID}-timings.txt"
METRICS="$RESULTS_DIR/${RUN_ID}-metrics.txt"
ALL_RESULTS="$RESULTS_DIR/${RUN_ID}-all_results.json"
# Kubernetes resource names must be RFC 1123 (lowercase) — keep RUN_ID upper for
# filenames but lowercase it for the PyTorchJob name and label selectors.
JOB_NAME="baseline-$(echo "$RUN_ID" | tr '[:upper:]' '[:lower:]')"
JOB_YAML="$RESULTS_DIR/${RUN_ID}-pytorchjob.yaml"

if [[ ! -f "$JOB_YAML" ]]; then
  echo "ERROR: $JOB_YAML not found. Save the PyTorchJob YAML first (see baseline.md Step 3)."
  exit 1
fi

# ── Ensure the timings file ends with a newline before we append ──────────────
# If the previous writer (baseline_pool.sh) was interrupted or the file was
# hand-edited, the last byte may not be \n, causing our first echo to run on.
if [[ -s "$TIMINGS" && -n "$(tail -c1 "$TIMINGS")" ]]; then
  echo "" >> "$TIMINGS"
fi

# ── Apply the job ─────────────────────────────────────────────────────────────
T_TRAIN_START=$(date +%s)
echo "t_training_start  = $(date +%T)" | tee -a "$TIMINGS"

kubectl apply -f "$JOB_YAML"

# ── Wait for master pod to be created (race fix + blip-resilient) ────────────
# Single loop that both polls and captures the name; transient kubectl errors
# leave MASTER_POD empty and the loop simply tries again.
echo "Waiting for master pod of $JOB_NAME..."
MASTER_POD=""
until [[ -n "$MASTER_POD" ]]; do
  MASTER_POD=$(kubectl get pods -n "$K8S_NAMESPACE" \
    -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=master \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  [[ -z "$MASTER_POD" ]] && sleep 2
done
echo "Master pod: $MASTER_POD"

# ── Poll for Succeeded (resilient to API server connection blips) ─────────────
echo "Polling for $JOB_NAME to Succeed (timeout 2h)..."
DEADLINE=$(( $(date +%s) + 7200 ))
while true; do
  if [[ $(date +%s) -gt $DEADLINE ]]; then
    echo "ERROR: timeout waiting for $JOB_NAME"
    exit 1
  fi
  SUCCEEDED=$(kubectl get pytorchjob "$JOB_NAME" -n "$K8S_NAMESPACE" \
    -o jsonpath='{.status.conditions[?(@.type=="Succeeded")].status}' 2>/dev/null || true)
  if [[ "$SUCCEEDED" == "True" ]]; then break; fi
  FAILED=$(kubectl get pytorchjob "$JOB_NAME" -n "$K8S_NAMESPACE" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
  if [[ "$FAILED" == "True" ]]; then
    echo "ERROR: PyTorchJob $JOB_NAME entered Failed state"
    exit 1
  fi
  sleep 10
done

T_TRAIN_END=$(date +%s)
APPLY_TO_SUCCEEDED_S=$((T_TRAIN_END - T_TRAIN_START))
echo "t_training_end    = $(date +%T)" | tee -a "$TIMINGS"
echo "apply_to_succeeded_s = ${APPLY_TO_SUCCEEDED_S}" | tee -a "$TIMINGS"

# Pull PyTorchJob's own start/completion timestamps so training_s matches the
# operator's distributedtraining_training_seconds window (pod runtime only).
PJ_START=$(kubectl get pytorchjob "$JOB_NAME" -n "$K8S_NAMESPACE" \
  -o jsonpath='{.status.startTime}' 2>/dev/null || true)
PJ_END=$(kubectl get pytorchjob "$JOB_NAME" -n "$K8S_NAMESPACE" \
  -o jsonpath='{.status.completionTime}' 2>/dev/null || true)

if [[ -n "$PJ_START" && -n "$PJ_END" ]]; then
  TRAINING_S=$(( $(date -d "$PJ_END" +%s) - $(date -d "$PJ_START" +%s) ))
  echo "pj_start_time     = $PJ_START" | tee -a "$TIMINGS"
  echo "pj_completion_time = $PJ_END" | tee -a "$TIMINGS"
  echo "training_s        = ${TRAINING_S}" | tee -a "$TIMINGS"
else
  echo "WARNING: PyTorchJob status timestamps missing; falling back to apply→Succeeded" >&2
  TRAINING_S=$APPLY_TO_SUCCEEDED_S
  echo "training_s        = ${TRAINING_S}  # FALLBACK: apply→Succeeded" | tee -a "$TIMINGS"
fi

# ── Collect results ───────────────────────────────────────────────────────────
T_COLLECT_START=$(date +%s)
echo "t_collection_start = $(date +%T)" | tee -a "$TIMINGS"

# Read from the NFS server pod (always Running) rather than the master pod
# (which transitions to Succeeded after training and refuses exec).
# Retry the lookup and the cp to survive transient API server connection blips
# (common on WSL2 long-running sessions).
NFS_POD=""
for i in 1 2 3 4 5 6; do
  NFS_POD=$(kubectl get pod -n "$K8S_NAMESPACE" -l role=nfs-server \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  [[ -n "$NFS_POD" ]] && break
  echo "NFS pod lookup failed (attempt $i/6); retrying in 10s..."
  sleep 10
done
if [[ -z "$NFS_POD" ]]; then
  echo "ERROR: could not find an NFS server pod after 6 attempts"
  exit 1
fi
echo "NFS pod: $NFS_POD"

for i in 1 2 3 4 5 6; do
  if kubectl cp "$K8S_NAMESPACE/${NFS_POD}:/exports/checkpoints/all_results.json" "$ALL_RESULTS"; then
    break
  fi
  if [[ $i -eq 6 ]]; then
    echo "ERROR: kubectl cp failed after 6 attempts"
    exit 1
  fi
  echo "kubectl cp failed (attempt $i/6); retrying in 10s..."
  sleep 10
done

# ── Parse training metrics ────────────────────────────────────────────────────
python3 - <<EOF | tee "$METRICS"
import json, math
r = json.load(open("$ALL_RESULTS"))
loss = r.get('eval_loss') or r.get('train_loss', 0)
ppl  = r.get('perplexity') or math.exp(loss)
sps  = r.get('train_samples_per_second', 0)
runtime = r.get('train_runtime', 0)
eval_runtime = r.get('eval_runtime', 0)
print(f'loss             = {loss:.4f}')
print(f'perplexity       = {ppl:.4f}')
print(f'samplesPerSecond = {sps:.4f}')
print(f'train_runtime    = {runtime:.1f}')
print(f'eval_runtime     = {eval_runtime:.1f}')
EOF

T_COLLECT_END=$(date +%s)
COLLECTION_S=$((T_COLLECT_END - T_COLLECT_START))
echo "t_collection_end  = $(date +%T)" | tee -a "$TIMINGS"
echo "collection_s      = ${COLLECTION_S}" | tee -a "$TIMINGS"

# Mirror train_runtime + eval_runtime into timings.txt as secondary records
RUNTIME=$(grep '^train_runtime' "$METRICS" | awk '{print $NF}')
EVAL_RUNTIME=$(grep '^eval_runtime' "$METRICS" | awk '{print $NF}')
echo "train_runtime_s   = ${RUNTIME}" | tee -a "$TIMINGS"
echo "eval_runtime_s    = ${EVAL_RUNTIME}" | tee -a "$TIMINGS"

# ── Print ready-to-paste push command ─────────────────────────────────────────
LOSS=$(grep '^loss' "$METRICS" | awk '{print $NF}')
PPL=$(grep '^perplexity' "$METRICS" | awk '{print $NF}')
SPS=$(grep '^samplesPerSecond' "$METRICS" | awk '{print $NF}')
PROV_S=$(grep '^provisioning_s' "$TIMINGS" | tail -1 | awk '{print $NF}')
NODES=$(grep -oE 'nnodes=[0-9]+' "$JOB_YAML" | head -1 | cut -d= -f2)

cat <<HINT

Done. Push to Grafana with:

./scripts/push_baseline.sh \\
  --run    ${JOB_NAME} \\
  --nodes  ${NODES} \\
  --prov   ${PROV_S:-<PROV_S>} \\
  --train  ${TRAINING_S} \\
  --loss   ${LOSS} \\
  --ppl    ${PPL} \\
  --sps    ${SPS} \\
  --namespace ${METRIC_NAMESPACE}
HINT
