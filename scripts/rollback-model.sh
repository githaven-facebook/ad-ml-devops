#!/usr/bin/env bash
# rollback-model.sh — Roll back a model deployment to the previous Helm revision.
# Usage: bash scripts/rollback-model.sh --model <name> --env <staging|production> [--revision <n>]

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME=""
ENVIRONMENT="staging"
REVISION="${REVISION:-0}"  # 0 = previous revision
NAMESPACE="${NAMESPACE:-ml-platform}"
HELM_TIMEOUT="${HELM_TIMEOUT:-5m}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_NAME="$2"; shift 2 ;;
    --env) ENVIRONMENT="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: $0 --model <name> [--env staging|production] [--revision <n>]" >&2
  exit 1
fi

RELEASE_NAME="ml-serving-${MODEL_NAME}"

echo "==> Rolling back ${RELEASE_NAME} in ${ENVIRONMENT} (namespace: ${NAMESPACE})"

# ── Show history ──────────────────────────────────────────────────────────────
echo ""
echo "==> Current Helm release history:"
helm history "${RELEASE_NAME}" -n "${NAMESPACE}" --max 10

# ── Confirm rollback ──────────────────────────────────────────────────────────
if [[ "${CI:-false}" != "true" ]]; then
  echo ""
  TARGET_DESC="previous revision"
  [[ "$REVISION" != "0" ]] && TARGET_DESC="revision ${REVISION}"
  read -rp "Confirm rollback of ${RELEASE_NAME} to ${TARGET_DESC}? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Rollback cancelled."
    exit 0
  fi
fi

# ── Execute rollback ──────────────────────────────────────────────────────────
echo ""
echo "==> Executing helm rollback..."
helm rollback "${RELEASE_NAME}" "${REVISION}" \
  -n "${NAMESPACE}" \
  --wait \
  --timeout "${HELM_TIMEOUT}" \
  --cleanup-on-fail

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying rollout after rollback..."
kubectl rollout status "deployment/${RELEASE_NAME}" \
  -n "${NAMESPACE}" \
  --timeout="${HELM_TIMEOUT}"

echo ""
echo "==> Post-rollback pod status:"
kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/name=ml-serving" \
  -o custom-columns='NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready,IMAGE:.spec.containers[0].image'

echo ""
echo "==> Rollback of ${RELEASE_NAME} complete."
