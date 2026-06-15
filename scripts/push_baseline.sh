#!/usr/bin/env bash
# push_baseline.sh — Push a completed baseline run's metrics into Prometheus
# via Pushgateway so they appear in the DistributedTraining Grafana dashboard.
#
# Usage:
#   ./scripts/push_baseline.sh \
#       --run      baseline-B-1a \
#       --nodes    1 \
#       --prov     312 \
#       --train    427 \
#       --loss     2.3141 \
#       --ppl      10.1178 \
#       --sps      1.2340 \
#       --namespace default
#
# All values are taken from results/baseline-run-<RUN>-all_results.json
# and your manually recorded timestamps (timings.txt for the run).
#
# The script port-forwards to the Pushgateway for the push, then exits.
# Prometheus will scrape the Pushgateway on its next 30s interval and the
# metrics will appear in Grafana automatically (no dashboard changes needed).
#
# To delete a pushed run from Pushgateway (e.g. to redo a run):
#   ./scripts/push_baseline.sh --delete --run baseline-B-1a --namespace default

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
RUN=""
NODES=""
PROV_S=""
TRAIN_S=""
LOSS=""
PPL=""
SPS=""
NAMESPACE="default"
DELETE=false
PUSHGATEWAY_SVC="pushgateway-prometheus-pushgateway"
PUSHGATEWAY_NS="monitoring"
LOCAL_PORT=9091

# ── argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)       RUN="$2";       shift 2 ;;
    --nodes)     NODES="$2";    shift 2 ;;
    --prov)      PROV_S="$2";   shift 2 ;;
    --train)     TRAIN_S="$2";  shift 2 ;;
    --loss)      LOSS="$2";     shift 2 ;;
    --ppl)       PPL="$2";      shift 2 ;;
    --sps)       SPS="$2";      shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --delete)    DELETE=true;   shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── validation ────────────────────────────────────────────────────────────────
if [[ -z "$RUN" ]]; then
  echo "ERROR: --run is required (e.g. --run baseline-B-1a)"
  exit 1
fi

if [[ "$DELETE" == false ]]; then
  for flag in NODES PROV_S TRAIN_S LOSS PPL SPS; do
    if [[ -z "${!flag}" ]]; then
      echo "ERROR: --${flag,,} is required for a push (use --delete to remove a run)"
      exit 1
    fi
  done
fi

# ── cost calculation ───────────────────────────────────────────────────────────
# c4-highcpu-8 = $0.42/hr  (matches machine-costs-configmap.yaml)
COST_PER_HOUR=0.42
if [[ "$DELETE" == false ]]; then
  COST=$(python3 -c "
nodes=${NODES}; prov=${PROV_S}; train=${TRAIN_S}; cph=${COST_PER_HOUR}
print(f'{nodes * cph * (prov + train) / 3600:.4f}')
")
  echo "Computed cost: \$${COST} USD"
fi

# ── port-forward to Pushgateway ───────────────────────────────────────────────
echo "Port-forwarding to ${PUSHGATEWAY_SVC} in namespace ${PUSHGATEWAY_NS}..."
kubectl port-forward \
  "svc/${PUSHGATEWAY_SVC}" \
  "${LOCAL_PORT}:9091" \
  -n "${PUSHGATEWAY_NS}" &
PF_PID=$!

# Give port-forward a moment to establish
sleep 2

# Ensure cleanup on exit
trap "kill ${PF_PID} 2>/dev/null || true" EXIT

PUSH_URL="http://localhost:${LOCAL_PORT}/metrics/job/distributedtraining-baseline/namespace/${NAMESPACE}/dj_name/${RUN}"

# ── delete path ───────────────────────────────────────────────────────────────
if [[ "$DELETE" == true ]]; then
  echo "Deleting metrics for run '${RUN}' from Pushgateway..."
  curl -s -X DELETE "${PUSH_URL}"
  echo "Done. Metrics for ${RUN} will disappear from Prometheus after the next scrape (~30s)."
  exit 0
fi

# ── push path ─────────────────────────────────────────────────────────────────
echo "Pushing metrics for run '${RUN}' (nodes=${NODES}, prov=${PROV_S}s, train=${TRAIN_S}s)..."

# Build the metric payload.
# Label scheme matches exactly what the operator emits (namespace + dj_name),
# so these metrics appear in all existing dashboard panels when selected.
PAYLOAD=$(cat <<EOF
# TYPE distributedtraining_nodes_provisioned gauge
distributedtraining_nodes_provisioned{namespace="${NAMESPACE}",dj_name="${RUN}"} ${NODES}
# TYPE distributedtraining_provisioning_seconds gauge
distributedtraining_provisioning_seconds{namespace="${NAMESPACE}",dj_name="${RUN}"} ${PROV_S}
# TYPE distributedtraining_training_seconds gauge
distributedtraining_training_seconds{namespace="${NAMESPACE}",dj_name="${RUN}"} ${TRAIN_S}
# TYPE distributedtraining_cost_usd_actual gauge
distributedtraining_cost_usd_actual{namespace="${NAMESPACE}",dj_name="${RUN}"} ${COST}
# TYPE distributedtraining_training_metric gauge
distributedtraining_training_metric{namespace="${NAMESPACE}",dj_name="${RUN}",metric="loss"} ${LOSS}
distributedtraining_training_metric{namespace="${NAMESPACE}",dj_name="${RUN}",metric="perplexity"} ${PPL}
distributedtraining_training_metric{namespace="${NAMESPACE}",dj_name="${RUN}",metric="samplesPerSecond"} ${SPS}
# TYPE distributedtraining_phase_transitions_total counter
# This synthetic counter is required so the dj_name variable query
# (label_values(distributedtraining_phase_transitions_total, dj_name)) includes
# baseline jobs in the Grafana dropdown.
distributedtraining_phase_transitions_total{namespace="${NAMESPACE}",dj_name="${RUN}",phase="Succeeded"} 1
EOF
)

HTTP_STATUS=$(echo "${PAYLOAD}" | curl -s -o /dev/null -w "%{http_code}" \
  --data-binary @- \
  "${PUSH_URL}")

if [[ "${HTTP_STATUS}" == "200" ]] || [[ "${HTTP_STATUS}" == "202" ]]; then
  echo "✓ Push succeeded (HTTP ${HTTP_STATUS})"
  echo ""
  echo "Summary for ${RUN}:"
  echo "  Nodes provisioned : ${NODES}"
  echo "  Provisioning time : ${PROV_S}s"
  echo "  Training time     : ${TRAIN_S}s"
  echo "  Total E2E (prov+train): $(( ${PROV_S} + ${TRAIN_S} ))s"
  echo "  Estimated cost    : \$${COST} USD"
  echo "  Loss              : ${LOSS}"
  echo "  Perplexity        : ${PPL}"
  echo "  Samples/sec       : ${SPS}"
  echo ""
  echo "Metrics will appear in Prometheus after the next scrape (~30s)."
  echo "Then open Grafana, select namespace='${NAMESPACE}' and dj_name='${RUN}'."
else
  echo "ERROR: Push failed (HTTP ${HTTP_STATUS})"
  exit 1
fi
