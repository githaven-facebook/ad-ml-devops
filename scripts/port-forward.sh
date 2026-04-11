#!/usr/bin/env bash
# port-forward.sh — Port forward Kubeflow UI, Grafana, and model serving endpoints.
# Usage: bash scripts/port-forward.sh [--service kubeflow|grafana|serving|all]

set -euo pipefail

SERVICE="${1:-all}"
KUBEFLOW_NS="${KUBEFLOW_NS:-kubeflow}"
MONITORING_NS="${MONITORING_NS:-monitoring}"
ML_NS="${ML_NS:-ml-platform}"

declare -A PIDS

cleanup() {
  echo ""
  echo "==> Stopping port forwards..."
  for svc in "${!PIDS[@]}"; do
    if kill "${PIDS[$svc]}" 2>/dev/null; then
      echo "  Stopped ${svc} (PID ${PIDS[$svc]})"
    fi
  done
  exit 0
}
trap cleanup SIGINT SIGTERM

start_forward() {
  local name="$1"
  local ns="$2"
  local resource="$3"
  local local_port="$4"
  local remote_port="$5"

  echo "==> Forwarding ${name}: localhost:${local_port} → ${resource}:${remote_port} (ns: ${ns})"
  kubectl port-forward -n "${ns}" "${resource}" "${local_port}:${remote_port}" &>/dev/null &
  PIDS["$name"]=$!
  sleep 1
  if ! kill -0 "${PIDS[$name]}" 2>/dev/null; then
    echo "  WARNING: Failed to start port forward for ${name}"
    unset PIDS["$name"]
  fi
}

echo "==> Starting port forwards (Ctrl+C to stop all)"
echo ""

case "$SERVICE" in
  kubeflow|all)
    start_forward "kubeflow-ui" "${KUBEFLOW_NS}" "svc/ml-pipeline-ui" 8080 80
    start_forward "kubeflow-api" "${KUBEFLOW_NS}" "svc/ml-pipeline" 8888 8888
    start_forward "mlflow" "${ML_NS}" "svc/mlflow" 5000 5000
    ;;
esac

case "$SERVICE" in
  grafana|all)
    start_forward "grafana" "${MONITORING_NS}" "svc/grafana" 3001 80
    start_forward "prometheus" "${MONITORING_NS}" "svc/prometheus-server" 9090 80
    ;;
esac

case "$SERVICE" in
  serving|all)
    start_forward "user-persona" "${ML_NS}" "svc/ml-serving-user-persona" 8001 80
    start_forward "autobid" "${ML_NS}" "svc/ml-serving-autobid" 8002 80
    ;;
esac

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "ERROR: No port forwards started. Check kubectl connectivity." >&2
  exit 1
fi

echo ""
echo "==> Active port forwards:"
echo ""
[[ -v PIDS[kubeflow-ui] ]]    && echo "  Kubeflow UI:      http://localhost:8080"
[[ -v PIDS[kubeflow-api] ]]   && echo "  Kubeflow API:     http://localhost:8888"
[[ -v PIDS[mlflow] ]]         && echo "  MLflow:           http://localhost:5000"
[[ -v PIDS[grafana] ]]        && echo "  Grafana:          http://localhost:3001"
[[ -v PIDS[prometheus] ]]     && echo "  Prometheus:       http://localhost:9090"
[[ -v PIDS[user-persona] ]]   && echo "  User Persona:     http://localhost:8001/predict"
[[ -v PIDS[autobid] ]]        && echo "  Autobid:          http://localhost:8002/predict"
echo ""
echo "  Press Ctrl+C to stop all port forwards."
echo ""

# Wait for all background processes
wait
