# ad-ml-devops

ML DevOps infrastructure for ad model training and serving on Kubernetes. This repository contains the platform engineering layer that powers training pipelines, model serving, and operational tooling for the **User Persona** and **Autobidding** models.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                      │
│  ci.yml → lint/test/helm-lint/docker-build                      │
│  train-pipeline.yml → scheduled + manual KFP triggers           │
│  deploy-model.yml → build bento → push image → helm upgrade     │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌───────────────┐
   │  Kubeflow   │ │  MLflow  │ │  Prometheus /  │
   │  Pipelines  │ │ Registry │ │    Grafana     │
   └──────┬──────┘ └────┬─────┘ └───────────────┘
          │             │
          ▼             ▼
   ┌─────────────────────────────┐
   │     ml-platform namespace   │
   │  ┌─────────────────────┐   │
   │  │  BentoML Serving     │   │
   │  │  user-persona-svc    │   │
   │  │  autobid-svc         │   │
   │  └──────────┬──────────┘   │
   │             │ Argo Rollouts │
   │             │ canary deploy │
   └─────────────┼───────────────┘
                 ▼
          Ingress / Istio
```

## Kubeflow Pipelines

### User Persona Pipeline (`pipelines/user_persona_pipeline.py`)

Runs **daily at 02:00 UTC**. DAG:

```
validate_data → train (4×A100 GPU, DDP) → evaluate → [if metrics pass] register_model → notify
```

Key parameters (see `pipelines/configs/user_persona_pipeline.yaml`):
- `promotion_metric`: `auc` — minimum 0.5% relative improvement required
- `significance_level`: 0.05 (paired t-test on bootstrap AUC distributions)
- `num_workers`: 4, `gpus_per_worker`: 4 (16 GPUs total)
- Promoted to **Staging** on pass; retained in MLflow but not promoted otherwise

### Autobid Pipeline (`pipelines/autobid_pipeline.py`)

Runs **every 6 hours**. DAG:

```
validate_data → train (8×A100 GPU, DDP) → evaluate → [if metrics pass] shadow_test → [if latency OK] register_model → notify
```

Key parameters (see `pipelines/configs/autobid_pipeline.yaml`):
- Stricter validation: `max_null_ratio=0.02`, `max_drift_pvalue=0.005`
- Promotion criteria: 1.0% min improvement, p < 0.01
- Shadow test: p99 inference latency must be ≤ 50ms
- Conservative because autobid affects ad spend directly

### Pipeline Components (`pipelines/components/`)

| Component | Description |
|-----------|-------------|
| `data_validation` | Schema check, null ratio, freshness, KS-test drift detection |
| `training` | Launches Kubernetes PyTorchJob (DDP), waits for completion, writes MLflow run ID |
| `evaluation` | Bootstrap AUC/NDCG comparison vs baseline, paired t-test significance |
| `model_registry` | Registers in MLflow Registry, transitions stage, tags with lineage metadata |
| `notification` | Slack Block Kit messages + PagerDuty v2 events on failure |

## BentoML Serving

### User Persona Service (`bentoml/user_persona_service.py`)

- **Endpoint**: `POST /predict` — batch user IDs → persona vectors (128-dim) + segment IDs + confidence scores
- **Runner**: MLflow-backed PyTorch runner with adaptive batching (max 512, 20ms window)
- **GPU**: 1×A100 per runner worker

### Autobid Service (`bentoml/autobid_service.py`)

- **Endpoints**: `POST /predict` (single), `POST /batch_predict` (batch)
- **Response cache**: TTL=5s, 10k entry LRU, keyed by SHA-256 of feature dict — reduces GPU pressure for repeat campaigns
- **Batch deduplication**: cache hits separated from misses before inference, merged back in original order
- **Bid multiplier**: clipped to [0.1, 10.0]

Configuration: `bentoml/configuration.yaml` — workers, timeout, batching, OTLP tracing, Prometheus metrics.

## Deployment Guide

### 1. Build and push a BentoML image

```bash
make build-bento MODEL_NAME=user-persona MODEL_VERSION=1.4.2
make push-bento MODEL_NAME=user-persona MODEL_VERSION=1.4.2
```

### 2. Deploy to staging (helm upgrade)

```bash
make deploy MODEL_NAME=user-persona IMAGE_TAG=1.4.2 ENVIRONMENT=staging
```

This runs `helm upgrade --install --atomic` with `values-staging.yaml` overrides. `--atomic` rolls back automatically on failure.

### 3. Canary deploy to production

Production uses Argo Rollouts for a phased canary:

```
10% → pause 10m → 30% → pause 10m → 50% → pause 10m → 100%
```

Analysis gates: Prometheus queries check p99 latency and error rate between steps. Rollout aborts and rolls back if either gate fails (`failureLimit: 3`).

```bash
make deploy MODEL_NAME=user-persona IMAGE_TAG=1.4.2 ENVIRONMENT=production
```

### 4. Rollback

```bash
make rollback MODEL_NAME=user-persona ENVIRONMENT=production
# Or to a specific revision:
bash scripts/rollback-model.sh --model user-persona --env production --revision 3
```

## Kubernetes Infrastructure

### Base (`k8s/base/`)

| Resource | Purpose |
|----------|---------|
| `namespace.yaml` | `ml-platform` with Istio injection + pod security standards |
| `gpu-node-pool.yaml` | RuntimeClass, tolerations for nvidia.com/gpu nodes |
| `resource-quotas.yaml` | Quota: 32 GPU, 256 CPU, 1Ti memory per namespace |
| `network-policies.yaml` | Default deny-all; allow serving ingress, DDP training ports, DNS, Prometheus scrape |

### Overlays

- **staging** (`k8s/overlays/staging/`): 4 GPU quota, 1 replica, caching disabled
- **production** (`k8s/overlays/production/`): Full quota, HPA on CPU/memory/p99 latency

## Helm Chart (`helm/ml-serving/`)

Templates: `Deployment`, `Service`, `HPA`, `PodDisruptionBudget`, `Ingress`, `ConfigMap`, `ServiceAccount`, `canary` (Argo Rollouts).

```bash
# Render templates locally
make helm-template ENVIRONMENT=staging

# Diff against running release
make helm-diff ENVIRONMENT=production
```

## Monitoring & Alerting

### Model Monitor (`monitoring/model_monitor.py`)

Scrapes Prometheus every 60s, computes p50/p95/p99 latency, error rate, throughput, and pushes to Pushgateway. Alerts fire via AlertManager rules.

### Data Drift Detector (`monitoring/data_drift_detector.py`)

Scheduled comparison of production serving samples against the training reference distribution:
- **PSI** (Population Stability Index): threshold 0.2
- **KL divergence**: threshold 0.5
- **JS divergence**: threshold 0.3

Autobid uses stricter thresholds (PSI 0.15, KL 0.3, JS 0.2) given financial impact.

### Prometheus Rules (`k8s/monitoring/`)

| Alert | Condition | Severity |
|-------|-----------|---------|
| `MLModelHighLatencyP99` | user-persona p99 > 100ms, autobid p99 > 50ms for 5m | warning |
| `MLModelCriticalLatencyP99` | user-persona p99 > 500ms, autobid p99 > 200ms for 2m | critical |
| `MLModelHighErrorRate` | error rate > 1% for 5m | warning |
| `MLModelCriticalErrorRate` | error rate > 5% for 2m | critical |
| `MLModelPredictionDrift` | PSI > 0.2 for 10m | warning |
| `MLGPUHighUtilization` | GPU > 95% for 10m | warning |
| `MLPipelineFailureRate` | pipeline success rate < 80% over 24h | warning |

Grafana dashboard ConfigMap: `k8s/monitoring/grafana-dashboards.yaml` (request rate, latency percentiles, error rate, GPU utilization, PSI, pipeline success rate).

## CI/CD Workflows

### `ci.yml` — Pull Request / Push

1. **lint**: flake8, black, isort, mypy
2. **test**: pytest (unit tests, no GPU required)
3. **helm-lint**: `helm lint --strict` for default/staging/production values
4. **docker-build**: builds `Dockerfile.pipeline` (no push)
5. **yaml-validate**: yamllint on K8s manifests and pipeline configs

### `train-pipeline.yml` — Training

Triggered by:
- Manual `workflow_dispatch` (pipeline, environment, version, wait flag)
- Schedule: user-persona daily 02:00 UTC, autobid every 6h

Injects Slack/PagerDuty secrets from GitHub environment secrets.

### `deploy-model.yml` — Model Deployment

Manual `workflow_dispatch` with inputs: model, version, environment, dry_run.

Steps: authenticate GCR → build+push bento → `helm upgrade --atomic` → `kubectl rollout status`.

## Runbook: Common Operations

### Pipeline stuck / failed

```bash
# Check KFP UI
bash scripts/port-forward.sh kubeflow
# Open http://localhost:8080

# Check PyTorchJob status
kubectl get pytorchjobs -n ml-platform
kubectl describe pytorchjob <job-name> -n ml-platform
kubectl logs -n ml-platform -l training-run=<job-name> --tail=100
```

### Serving latency spike

```bash
# Check Grafana
bash scripts/port-forward.sh grafana
# Open http://localhost:3001

# Check pod resource usage
kubectl top pods -n ml-platform -l app.kubernetes.io/component=ml-serving

# Check GPU utilization
kubectl exec -n ml-platform <pod> -- nvidia-smi
```

### Emergency rollback

```bash
bash scripts/rollback-model.sh --model autobid --env production
# Confirms before executing; CI=true skips prompt
```

### Force-promote model to production

```bash
python3 - <<'EOF'
import mlflow
client = mlflow.tracking.MlflowClient("http://mlflow.ml-platform.svc.cluster.local:5000")
client.transition_model_version_stage("autobid", version="42", stage="Production", archive_existing_versions=True)
EOF
```

## Local Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
make lint

# Run tests
make test

# Compile pipelines locally
make build-pipeline

# Port-forward all services
bash scripts/port-forward.sh
```

## Repository Structure

```
ad-ml-devops/
├── pipelines/          # KFP pipeline definitions and components
├── bentoml/            # BentoML service definitions and config
├── k8s/                # Kubernetes manifests (base + overlays + monitoring)
├── helm/ml-serving/    # Helm chart for model serving
├── docker/             # Multi-stage Dockerfiles
├── scripts/            # Operational shell scripts
├── monitoring/         # Model monitor and data drift detector
└── tests/              # pytest test suite
```
