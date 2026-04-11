.PHONY: setup-kubeflow build-pipeline run-pipeline build-bento deploy test lint fmt clean

KUBEFLOW_NAMESPACE ?= kubeflow
ML_NAMESPACE ?= ml-platform
PIPELINE_HOST ?= http://localhost:8080
MODEL_NAME ?= user-persona
MODEL_VERSION ?= latest
ENVIRONMENT ?= staging
IMAGE_REGISTRY ?= gcr.io/fb-ads-ml
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)

# ── Setup ──────────────────────────────────────────────────────────────────────

setup-kubeflow:
	@echo "Installing Kubeflow Pipelines..."
	bash scripts/setup-kubeflow.sh

# ── Pipelines ─────────────────────────────────────────────────────────────────

build-pipeline:
	@echo "Compiling KFP pipelines..."
	python -m pipelines.user_persona_pipeline
	python -m pipelines.autobid_pipeline
	@echo "Pipelines compiled to pipeline_artifacts/"

run-pipeline:
	@echo "Triggering pipeline: $(MODEL_NAME)"
	bash scripts/run-pipeline.sh \
		--pipeline $(MODEL_NAME) \
		--host $(PIPELINE_HOST) \
		--env $(ENVIRONMENT)

# ── BentoML ───────────────────────────────────────────────────────────────────

build-bento:
	@echo "Building BentoML bento for $(MODEL_NAME)..."
	cd bentoml && bentoml build -f bentofile.yaml

push-bento:
	@echo "Pushing bento image $(MODEL_NAME):$(MODEL_VERSION)..."
	bentoml containerize $(MODEL_NAME):$(MODEL_VERSION) \
		--image-tag $(IMAGE_REGISTRY)/$(MODEL_NAME):$(IMAGE_TAG)
	docker push $(IMAGE_REGISTRY)/$(MODEL_NAME):$(IMAGE_TAG)

# ── Deploy ────────────────────────────────────────────────────────────────────

deploy:
	@echo "Deploying $(MODEL_NAME) to $(ENVIRONMENT)..."
	bash scripts/deploy-model.sh \
		--model $(MODEL_NAME) \
		--version $(IMAGE_TAG) \
		--env $(ENVIRONMENT)

rollback:
	@echo "Rolling back $(MODEL_NAME) in $(ENVIRONMENT)..."
	bash scripts/rollback-model.sh \
		--model $(MODEL_NAME) \
		--env $(ENVIRONMENT)

# ── K8s ───────────────────────────────────────────────────────────────────────

apply-base:
	kubectl apply -k k8s/base/

apply-staging:
	kubectl apply -k k8s/overlays/staging/

apply-production:
	kubectl apply -k k8s/overlays/production/

apply-monitoring:
	kubectl apply -f k8s/monitoring/

port-forward:
	bash scripts/port-forward.sh

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-pipelines:
	pytest tests/test_pipelines.py -v --tb=short

test-bentoml:
	pytest tests/test_bentoml_services.py -v --tb=short

test-helm:
	pytest tests/test_helm_charts.py -v --tb=short

# ── Lint / Format ─────────────────────────────────────────────────────────────

lint:
	flake8 pipelines/ monitoring/ tests/
	mypy pipelines/ monitoring/
	helm lint helm/ml-serving/

fmt:
	black pipelines/ monitoring/ tests/
	isort pipelines/ monitoring/ tests/

# ── Helm ──────────────────────────────────────────────────────────────────────

helm-template:
	helm template ml-serving helm/ml-serving/ \
		-f helm/ml-serving/values-$(ENVIRONMENT).yaml \
		--debug

helm-diff:
	helm diff upgrade ml-serving helm/ml-serving/ \
		-f helm/ml-serving/values-$(ENVIRONMENT).yaml \
		-n $(ML_NAMESPACE)

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ .coverage dist/ build/
	rm -rf pipeline_artifacts/*.yaml 2>/dev/null || true
	@echo "Clean complete."
