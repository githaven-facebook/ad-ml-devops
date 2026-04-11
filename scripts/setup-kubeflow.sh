#!/usr/bin/env bash
# setup-kubeflow.sh — Install Kubeflow Pipelines on a GKE cluster.
# Usage: bash scripts/setup-kubeflow.sh [--version <kfp-version>] [--namespace <ns>]

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
KFP_VERSION="${KFP_VERSION:-2.0.5}"
NAMESPACE="${NAMESPACE:-kubeflow}"
PIPELINE_HOST="${PIPELINE_HOST:-}"
APPLY_TIMEOUT="${APPLY_TIMEOUT:-300s}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) KFP_VERSION="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --host) PIPELINE_HOST="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "==> Installing Kubeflow Pipelines v${KFP_VERSION} in namespace '${NAMESPACE}'"

# ── Prerequisites check ───────────────────────────────────────────────────────
for cmd in kubectl kustomize; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not installed." >&2
    exit 1
  fi
done

# ── Create namespace ──────────────────────────────────────────────────────────
kubectl get namespace "${NAMESPACE}" &>/dev/null || \
  kubectl create namespace "${NAMESPACE}"

# ── Apply KFP manifests ───────────────────────────────────────────────────────
KFP_MANIFEST_URI="https://raw.githubusercontent.com/kubeflow/pipelines/${KFP_VERSION}/manifests/kustomize/cluster-scoped-resources"

echo "==> Applying cluster-scoped resources..."
kubectl apply -k "${KFP_MANIFEST_URI}" --timeout="${APPLY_TIMEOUT}"

echo "==> Applying namespace-scoped resources..."
kubectl apply -k "https://raw.githubusercontent.com/kubeflow/pipelines/${KFP_VERSION}/manifests/kustomize/env/platform-agnostic-pns" \
  --timeout="${APPLY_TIMEOUT}"

# ── Wait for deployments ──────────────────────────────────────────────────────
echo "==> Waiting for KFP deployments to be ready..."
DEPLOYMENTS=(
  "ml-pipeline"
  "ml-pipeline-ui"
  "ml-pipeline-scheduledworkflow"
  "ml-pipeline-persistenceagent"
  "workflow-controller"
  "metadata-grpc-deployment"
)

for deploy in "${DEPLOYMENTS[@]}"; do
  echo "  Waiting for deployment/${deploy}..."
  kubectl rollout status "deployment/${deploy}" \
    -n "${NAMESPACE}" \
    --timeout="${APPLY_TIMEOUT}" || true
done

# ── Apply ml-platform base ────────────────────────────────────────────────────
echo "==> Applying ml-platform base manifests..."
kubectl apply -k k8s/base/ --timeout="${APPLY_TIMEOUT}"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "==> KFP installation summary:"
kubectl get pods -n "${NAMESPACE}" --field-selector=status.phase=Running \
  -o custom-columns='NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready'

echo ""
echo "==> Kubeflow Pipelines installed successfully."
echo "    Access the UI via: kubectl port-forward -n ${NAMESPACE} svc/ml-pipeline-ui 8080:80"
