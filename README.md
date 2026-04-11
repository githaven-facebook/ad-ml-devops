# ad-ml-devops

ML DevOps infrastructure for ad model training and serving on Kubernetes. This repository contains the platform engineering layer that powers training pipelines, model serving, and operational tooling for the User Persona and Autobidding models.

## Overview

- **Kubeflow Pipelines** — Orchestrates model training workflows with DAG-based pipelines
- **BentoML** — Packages and serves models with adaptive batching and GPU inference
- **Kubernetes / Helm** — Manages serving infrastructure with canary deployments and autoscaling
- **Prometheus / Grafana** — Monitors model performance, data drift, and infrastructure health
