#!/usr/bin/env bash
# deploy-model.sh — Build BentoML bento, push image, and helm upgrade with canary.
# Usage: bash scripts/deploy-model.sh --model <name> --version <tag> --env <staging|production>

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME=""
IMAGE_TAG=""
ENVIRONMENT="staging"
REGISTRY="${REGISTRY:-gcr.io/fb-ads-ml}"
NAMESPACE="${NAMESPACE:-ml-platform}"
HELM_TIMEOUT="${HELM_TIMEOUT:-10m}"
DRY_RUN="${DRY_RUN:-false}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_NAME="$2"; shift 2 ;;
    --version) IMAGE_TAG="$2"; shift 2 ;;
    --env) ENVIRONMENT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_NAME" || -z "$IMAGE_TAG" ]]; then
  echo "Usage: $0 --model <name> --version <tag> [--env staging|production] [--dry-run]" >&2
  exit 1
fi

RELEASE_NAME="ml-serving-${MODEL_NAME}"
IMAGE_REF="${REGISTRY}/${MODEL_NAME}:${IMAGE_TAG}"
VALUES_FILE="helm/ml-serving/values-${ENVIRONMENT}.yaml"

echo "==> Deploying ${MODEL_NAME}:${IMAGE_TAG} to ${ENVIRONMENT}"
echo "    Image:   ${IMAGE_REF}"
echo "    Release: ${RELEASE_NAME}"
echo "    Env:     ${ENVIRONMENT}"

# ── Prerequisites check ───────────────────────────────────────────────────────
for cmd in docker helm kubectl bentoml; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not installed." >&2
    exit 1
  fi
done

if [[ ! -f "$VALUES_FILE" ]]; then
  echo "ERROR: Values file not found: ${VALUES_FILE}" >&2
  exit 1
fi

# ── Build BentoML bento ───────────────────────────────────────────────────────
echo ""
echo "==> Building BentoML bento for ${MODEL_NAME}..."
pushd bentoml > /dev/null
bentoml build -f bentofile.yaml --version "${IMAGE_TAG}"
popd > /dev/null

# ── Containerize bento ───────────────────────────────────────────────────────
echo ""
echo "==> Containerizing bento → ${IMAGE_REF}..."
bentoml containerize "${MODEL_NAME}:${IMAGE_TAG}" --image-tag "${IMAGE_REF}"

# ── Push image ────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" != "true" ]]; then
  echo ""
  echo "==> Pushing image ${IMAGE_REF}..."
  docker push "${IMAGE_REF}"
else
  echo "==> [DRY RUN] Skipping docker push"
fi

# ── Helm upgrade ──────────────────────────────────────────────────────────────
echo ""
echo "==> Running helm upgrade for ${RELEASE_NAME}..."

HELM_ARGS=(
  upgrade --install "${RELEASE_NAME}" helm/ml-serving/
  --namespace "${NAMESPACE}"
  --create-namespace
  --values "helm/ml-serving/values.yaml"
  --values "${VALUES_FILE}"
  --set "image.tag=${IMAGE_TAG}"
  --set "model.name=${MODEL_NAME}"
  --timeout "${HELM_TIMEOUT}"
  --atomic
  --cleanup-on-fail
  --history-max 5
)

if [[ "$DRY_RUN" == "true" ]]; then
  HELM_ARGS+=(--dry-run --debug)
fi

helm "${HELM_ARGS[@]}"

# ── Verify rollout ────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" != "true" ]]; then
  echo ""
  echo "==> Verifying rollout..."
  kubectl rollout status "deployment/${RELEASE_NAME}" \
    -n "${NAMESPACE}" \
    --timeout="${HELM_TIMEOUT}"

  echo ""
  echo "==> Deployment complete."
  kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/name=ml-serving" \
    -o custom-columns='NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready,IMAGE:.spec.containers[0].image'
fi

echo ""
echo "==> ${MODEL_NAME}:${IMAGE_TAG} deployed to ${ENVIRONMENT} successfully."
