#!/usr/bin/env bash
# run-pipeline.sh — Trigger a Kubeflow pipeline run with parameters.
# Usage: bash scripts/run-pipeline.sh --pipeline <user-persona|autobid> --host <kfp-host> --env <staging|production>

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PIPELINE_NAME=""
KFP_HOST="${KFP_HOST:-http://localhost:8080}"
ENVIRONMENT="${ENVIRONMENT:-staging}"
MODEL_VERSION="${MODEL_VERSION:-$(date +%Y%m%d-%H%M%S)}"
DATASET_VERSION="${DATASET_VERSION:-$(date +%Y%m%d)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
WAIT="${WAIT:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-7200}"  # 2 hours

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline) PIPELINE_NAME="$2"; shift 2 ;;
    --host) KFP_HOST="$2"; shift 2 ;;
    --env) ENVIRONMENT="$2"; shift 2 ;;
    --model-version) MODEL_VERSION="$2"; shift 2 ;;
    --dataset-version) DATASET_VERSION="$2"; shift 2 ;;
    --wait) WAIT="true"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$PIPELINE_NAME" ]]; then
  echo "Usage: $0 --pipeline <user-persona|autobid> [options]" >&2
  echo "Options:" >&2
  echo "  --host <url>             KFP API host (default: http://localhost:8080)" >&2
  echo "  --env <staging|production>" >&2
  echo "  --model-version <version>" >&2
  echo "  --dataset-version <version>" >&2
  echo "  --wait                   Wait for run completion" >&2
  exit 1
fi

CONFIG_FILE="pipelines/configs/${PIPELINE_NAME}_pipeline.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Pipeline config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

[[ -z "$EXPERIMENT_NAME" ]] && EXPERIMENT_NAME="${PIPELINE_NAME}-${ENVIRONMENT}"

echo "==> Triggering pipeline: ${PIPELINE_NAME}"
echo "    Host:            ${KFP_HOST}"
echo "    Environment:     ${ENVIRONMENT}"
echo "    Model version:   ${MODEL_VERSION}"
echo "    Dataset version: ${DATASET_VERSION}"
echo "    Experiment:      ${EXPERIMENT_NAME}"

# ── Load Slack/PD secrets from K8s ───────────────────────────────────────────
SLACK_WEBHOOK_URL=$(kubectl get secret ml-pipeline-secrets -n ml-platform \
  -o jsonpath='{.data.slack-webhook-url}' 2>/dev/null | base64 -d || echo "")

PD_ROUTING_KEY=$(kubectl get secret ml-pipeline-secrets -n ml-platform \
  -o jsonpath='{.data.pagerduty-routing-key}' 2>/dev/null | base64 -d || echo "")

# ── Submit pipeline run ───────────────────────────────────────────────────────
python3 - <<EOF
import kfp
import yaml

client = kfp.Client(host="${KFP_HOST}")

with open("${CONFIG_FILE}") as f:
    config = yaml.safe_load(f)

params = config.get("parameters", {})
params["model_version"] = "${MODEL_VERSION}"
params["dataset_version"] = "${DATASET_VERSION}"
params["slack_webhook_url"] = "${SLACK_WEBHOOK_URL}"
params["pagerduty_routing_key"] = "${PD_ROUTING_KEY}"

# Ensure experiment exists
try:
    experiment = client.get_experiment(experiment_name="${EXPERIMENT_NAME}")
except Exception:
    experiment = client.create_experiment("${EXPERIMENT_NAME}")

# Get or upload pipeline
pipeline_name = config["pipeline_name"]
try:
    pipeline = client.get_pipeline_id(pipeline_name)
except Exception:
    pipeline = None

run = client.create_run_from_pipeline_func(
    pipeline_func=None,
    run_name=f"${PIPELINE_NAME}-${MODEL_VERSION}",
    experiment_name="${EXPERIMENT_NAME}",
    arguments=params,
    pipeline_id=pipeline,
)

print(f"Pipeline run submitted: {run.run_id}")
print(f"View at: ${KFP_HOST}/#/runs/details/{run.run_id}")
EOF

echo ""
if [[ "$WAIT" == "true" ]]; then
  echo "==> Waiting for pipeline run to complete (timeout: ${WAIT_TIMEOUT}s)..."
  python3 - <<EOF
import kfp, sys, time

client = kfp.Client(host="${KFP_HOST}")
# Get the most recent run for this pipeline
runs = client.list_runs(
    experiment_id=client.get_experiment(experiment_name="${EXPERIMENT_NAME}").id,
    sort_by="created_at desc",
    page_size=1,
)
run_id = runs.runs[0].id
start = time.time()
while time.time() - start < ${WAIT_TIMEOUT}:
    run_detail = client.get_run(run_id)
    status = run_detail.run.status
    print(f"Status: {status} ({int(time.time() - start)}s elapsed)")
    if status in ("Succeeded", "Failed", "Skipped", "Error"):
        sys.exit(0 if status == "Succeeded" else 1)
    time.sleep(30)
print("Timeout waiting for pipeline run")
sys.exit(1)
EOF
fi

echo "==> Pipeline triggered successfully."
