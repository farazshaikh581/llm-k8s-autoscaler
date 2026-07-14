#!/bin/bash
# Pin the richer real-cluster experiment (workload-cpu) to dedicated worker
# node(s), isolated from other projects (blackbox, hv-phase0).
#
# Run this AFTER you have joined the new worker node(s) to the k3s cluster.
# It labels + taints the dedicated node(s) so only this experiment lands there,
# then patches workload-cpu with the matching nodeSelector + toleration.
#
# Usage:
#   bash setup_dedicated_nodes.sh <node-name> [<node-name> ...]
# Example:
#   bash setup_dedicated_nodes.sh k8s-worker3 k8s-worker4
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <node-name> [<node-name> ...]"; exit 1
fi

LABEL_KEY="autoscale-exp"
TAINT="dedicated=autoscale-exp:NoSchedule"
DEPLOY="workload-cpu"

echo "Dedicated nodes: $*"
for node in "$@"; do
    kubectl get node "$node" >/dev/null || { echo "ERROR: node $node not found (joined to cluster?)"; exit 1; }
    echo "  labeling + tainting $node"
    kubectl label   node "$node" "${LABEL_KEY}=true" --overwrite
    kubectl taint   node "$node" "$TAINT" --overwrite
done

echo "Patching $DEPLOY: nodeSelector ${LABEL_KEY}=true + toleration"
kubectl patch deployment "$DEPLOY" --type merge -p "$(cat <<JSON
{"spec":{"template":{"spec":{
  "nodeSelector":{"${LABEL_KEY}":"true"},
  "tolerations":[{"key":"dedicated","operator":"Equal","value":"autoscale-exp","effect":"NoSchedule"}]
}}}}
JSON
)"

echo "Waiting for $DEPLOY to reschedule onto the dedicated node(s)..."
kubectl rollout status deployment "$DEPLOY" --timeout=120s

echo
echo "=== verification ==="
echo "workload-cpu pods now on:"
kubectl get pods -l app="$DEPLOY" -o wide 2>/dev/null | awk 'NR==1||/workload-cpu/{print $1, $3, $7}' \
  || kubectl get pods -o wide | grep "$DEPLOY"
echo
echo "dedicated node allocatable CPU (need >= 5000m for cap-20 requests):"
for node in "$@"; do
    kubectl get node "$node" -o jsonpath="{.metadata.name}: {.status.allocatable.cpu} cpu{'\n'}"
done
echo
echo "Confirm other projects are NOT on the dedicated node(s):"
for node in "$@"; do
    other=$(kubectl get pods -A -o wide 2>/dev/null | awk -v n="$node" '$8==n && $1 ~ /blackbox|hv-phase0/{print $1"/"$2}')
    echo "  $node: ${other:-clean (no blackbox/hv-phase0 pods)}"
done
echo
echo "Done. If pods are on the dedicated node(s) and allocatable >= 5 cpu, launch:"
echo "  nohup setsid bash run_richer_real.sh >> logs/richer_real_main.log 2>&1 &"
