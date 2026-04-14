"""Tests for Helm chart template rendering and value validation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

CHART_DIR = Path(__file__).parent.parent / "helm" / "ml-serving"
VALUES_FILE = CHART_DIR / "values.yaml"
VALUES_STAGING = CHART_DIR / "values-staging.yaml"
VALUES_PRODUCTION = CHART_DIR / "values-production.yaml"


def _helm_available() -> bool:
    try:
        result = subprocess.run(["helm", "version", "--short"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _helm_template(extra_values: str | None = None, set_args: list[str] | None = None) -> dict:
    """Run `helm template` and return parsed YAML documents as a list."""
    cmd = [
        "helm", "template", "test-release", str(CHART_DIR),
        "--namespace", "ml-platform",
        "--values", str(VALUES_FILE),
    ]
    if extra_values:
        cmd += ["--values", extra_values]
    if set_args:
        for arg in set_args:
            cmd += ["--set", arg]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        pytest.fail(f"helm template failed:\n{result.stderr}")

    docs = list(yaml.safe_load_all(result.stdout))
    return {
        f"{d['kind']}/{d['metadata']['name']}": d
        for d in docs
        if d is not None and "kind" in d
    }


class TestValuesFiles:
    """Validate values YAML files are well-formed and contain required fields."""

    @pytest.fixture(params=["values.yaml", "values-staging.yaml", "values-production.yaml"])
    def values_path(self, request: pytest.FixtureRequest) -> Path:
        return CHART_DIR / request.param

    def test_values_file_is_valid_yaml(self, values_path: Path) -> None:
        assert values_path.exists(), f"Values file not found: {values_path}"
        with open(values_path) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert isinstance(data, dict)

    def test_default_values_has_required_keys(self) -> None:
        with open(VALUES_FILE) as f:
            values = yaml.safe_load(f)

        required_keys = [
            "image", "replicaCount", "service", "ingress",
            "resources", "autoscaling", "model", "canary",
        ]
        for key in required_keys:
            assert key in values, f"Missing required key: {key}"

    def test_production_has_more_replicas_than_staging(self) -> None:
        with open(VALUES_STAGING) as f:
            staging = yaml.safe_load(f)
        with open(VALUES_PRODUCTION) as f:
            production = yaml.safe_load(f)

        assert production["replicaCount"] > staging["replicaCount"]

    def test_production_canary_enabled(self) -> None:
        with open(VALUES_PRODUCTION) as f:
            production = yaml.safe_load(f)
        assert production["canary"]["enabled"] is True

    def test_staging_canary_disabled(self) -> None:
        with open(VALUES_STAGING) as f:
            staging = yaml.safe_load(f)
        assert staging["canary"]["enabled"] is False

    def test_production_has_topology_spread(self) -> None:
        with open(VALUES_PRODUCTION) as f:
            production = yaml.safe_load(f)
        assert "topologySpreadConstraints" in production
        assert len(production["topologySpreadConstraints"]) > 0

    def test_production_pdb_enabled(self) -> None:
        with open(VALUES_PRODUCTION) as f:
            production = yaml.safe_load(f)
        assert production["podDisruptionBudget"]["enabled"] is True
        assert production["podDisruptionBudget"]["minAvailable"] >= 2


class TestHelmTemplates:
    """Tests for rendered Helm template output."""

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_default_template_renders(self) -> None:
        """Default values should produce renderable templates."""
        resources = _helm_template()
        assert len(resources) > 0

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_deployment_resource_exists(self) -> None:
        resources = _helm_template()
        deployments = [k for k in resources if k.startswith("Deployment/")]
        assert len(deployments) > 0

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_service_resource_exists(self) -> None:
        resources = _helm_template()
        services = [k for k in resources if k.startswith("Service/")]
        assert len(services) > 0

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_hpa_rendered_when_autoscaling_enabled(self) -> None:
        resources = _helm_template()
        hpas = [k for k in resources if k.startswith("HorizontalPodAutoscaler/")]
        assert len(hpas) > 0

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_pdb_rendered_when_enabled(self) -> None:
        resources = _helm_template()
        pdbs = [k for k in resources if k.startswith("PodDisruptionBudget/")]
        assert len(pdbs) > 0

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_deployment_has_liveness_probe(self) -> None:
        resources = _helm_template()
        deployment = next((v for k, v in resources.items() if k.startswith("Deployment/")), None)
        assert deployment is not None
        containers = deployment["spec"]["template"]["spec"]["containers"]
        assert len(containers) > 0
        for container in containers:
            assert "livenessProbe" in container, "Deployment container missing livenessProbe"

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_deployment_has_readiness_probe(self) -> None:
        resources = _helm_template()
        deployment = next((v for k, v in resources.items() if k.startswith("Deployment/")), None)
        assert deployment is not None
        containers = deployment["spec"]["template"]["spec"]["containers"]
        for container in containers:
            assert "readinessProbe" in container, "Deployment container missing readinessProbe"

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_deployment_non_root_security_context(self) -> None:
        resources = _helm_template()
        deployment = next((v for k, v in resources.items() if k.startswith("Deployment/")), None)
        assert deployment is not None
        pod_spec = deployment["spec"]["template"]["spec"]
        security_ctx = pod_spec.get("securityContext", {})
        assert security_ctx.get("runAsNonRoot") is True

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_service_account_rendered(self) -> None:
        resources = _helm_template()
        sas = [k for k in resources if k.startswith("ServiceAccount/")]
        assert len(sas) > 0

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_staging_values_reduce_replicas(self) -> None:
        resources = _helm_template(extra_values=str(VALUES_STAGING))
        # With autoscaling enabled the replica count isn't set in the template body,
        # but the HPA minReplicas should be lower than production
        hpa = next((v for k, v in resources.items() if k.startswith("HorizontalPodAutoscaler/")), None)
        if hpa:
            assert hpa["spec"]["minReplicas"] <= 2

    @pytest.mark.skipif(not _helm_available(), reason="helm not installed")
    def test_production_values_enable_pdb(self) -> None:
        resources = _helm_template(extra_values=str(VALUES_PRODUCTION))
        pdbs = [k for k in resources if k.startswith("PodDisruptionBudget/")]
        assert len(pdbs) > 0


class TestChartMetadata:
    """Validate Chart.yaml content."""

    def test_chart_yaml_valid(self) -> None:
        chart_path = CHART_DIR / "Chart.yaml"
        assert chart_path.exists()
        with open(chart_path) as f:
            chart = yaml.safe_load(f)
        assert chart["apiVersion"] == "v2"
        assert "name" in chart
        assert "version" in chart
        assert "appVersion" in chart

    def test_chart_has_dependencies(self) -> None:
        with open(CHART_DIR / "Chart.yaml") as f:
            chart = yaml.safe_load(f)
        assert "dependencies" in chart
        dep_names = [d["name"] for d in chart["dependencies"]]
        assert "prometheus" in dep_names
        assert "grafana" in dep_names
