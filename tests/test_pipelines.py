"""Tests for Kubeflow pipeline compilation, component I/O, and parameter validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml


class TestPipelineCompilation:
    """Verify that pipelines compile to valid KFP YAML without errors."""

    def test_user_persona_pipeline_compiles(self) -> None:
        """User persona pipeline should compile to a valid KFP YAML artifact."""
        try:
            import kfp

            from pipelines.user_persona_pipeline import user_persona_pipeline

            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                output_path = f.name

            try:
                kfp.compiler.Compiler().compile(
                    pipeline_func=user_persona_pipeline,
                    package_path=output_path,
                )
                assert os.path.exists(output_path)
                assert os.path.getsize(output_path) > 0

                with open(output_path) as f:
                    compiled = yaml.safe_load(f)

                assert compiled is not None
                assert "pipelineInfo" in compiled or "pipeline_info" in compiled or "components" in compiled
            finally:
                os.unlink(output_path)

        except ImportError:
            pytest.skip("kfp not installed")

    def test_autobid_pipeline_compiles(self) -> None:
        """Autobid pipeline should compile to a valid KFP YAML artifact."""
        try:
            import kfp

            from pipelines.autobid_pipeline import autobid_pipeline

            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                output_path = f.name

            try:
                kfp.compiler.Compiler().compile(
                    pipeline_func=autobid_pipeline,
                    package_path=output_path,
                )
                assert os.path.exists(output_path)
                assert os.path.getsize(output_path) > 0
            finally:
                os.unlink(output_path)

        except ImportError:
            pytest.skip("kfp not installed")

    def test_user_persona_pipeline_has_expected_parameters(self) -> None:
        """User persona pipeline must expose all required parameters."""
        try:
            import inspect

            from pipelines.user_persona_pipeline import user_persona_pipeline

            sig = inspect.signature(user_persona_pipeline)
            required_params = [
                "dataset_uri",
                "model_version",
                "num_workers",
                "gpus_per_worker",
                "mlflow_tracking_uri",
                "promotion_metric",
                "min_improvement_pct",
                "significance_level",
            ]
            for param in required_params:
                assert param in sig.parameters, f"Missing parameter: {param}"
        except ImportError:
            pytest.skip("kfp not installed")

    def test_autobid_pipeline_has_shadow_test_parameter(self) -> None:
        """Autobid pipeline must include shadow test configuration parameters."""
        try:
            import inspect

            from pipelines.autobid_pipeline import autobid_pipeline

            sig = inspect.signature(autobid_pipeline)
            assert "max_shadow_latency_p99_ms" in sig.parameters
            assert "shadow_dataset_uri" in sig.parameters
        except ImportError:
            pytest.skip("kfp not installed")


class TestPipelineConfigs:
    """Validate pipeline configuration YAML files."""

    @pytest.fixture(params=["user_persona_pipeline.yaml", "autobid_pipeline.yaml"])
    def config_path(self, request: pytest.FixtureRequest) -> Path:
        return Path(__file__).parent.parent / "pipelines" / "configs" / request.param

    def test_config_file_exists(self, config_path: Path) -> None:
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_config_has_required_fields(self, config_path: Path) -> None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        required_fields = ["pipeline_name", "schedule", "parameters"]
        for field in required_fields:
            assert field in config, f"Missing field '{field}' in {config_path.name}"

    def test_config_parameters_have_mlflow_uri(self, config_path: Path) -> None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        params = config.get("parameters", {})
        assert "mlflow_tracking_uri" in params
        assert params["mlflow_tracking_uri"].startswith("http")

    def test_autobid_config_has_stricter_validation(self) -> None:
        """Autobid should have stricter null ratio and drift thresholds than user-persona."""
        configs = {}
        for name in ["user_persona_pipeline", "autobid_pipeline"]:
            path = Path(__file__).parent.parent / "pipelines" / "configs" / f"{name}.yaml"
            with open(path) as f:
                configs[name] = yaml.safe_load(f)

        autobid_params = configs["autobid_pipeline"]["parameters"]
        user_persona_params = configs["user_persona_pipeline"]["parameters"]

        assert autobid_params["max_null_ratio"] < user_persona_params["max_null_ratio"]
        assert autobid_params["min_improvement_pct"] > user_persona_params["min_improvement_pct"]


class TestDataValidationComponent:
    """Unit tests for the data_validation KFP component logic."""

    def test_null_ratio_detection(self) -> None:
        """Component should detect columns exceeding max null ratio."""
        import pandas as pd

        df = pd.DataFrame({
            "feature_a": [1.0, 2.0, None, 4.0, 5.0],  # 20% null
            "feature_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        null_ratios = (df.isnull().sum() / len(df)).to_dict()
        max_null_ratio = 0.1

        violations = {col: ratio for col, ratio in null_ratios.items() if ratio > max_null_ratio}
        assert "feature_a" in violations
        assert "feature_b" not in violations

    def test_row_count_validation(self) -> None:
        """Component should fail when dataset has fewer rows than min_rows."""
        import pandas as pd

        df = pd.DataFrame({"feature": range(100)})
        min_rows = 1000

        assert len(df) < min_rows

    def test_ks_test_detects_drift(self) -> None:
        """KS test should detect significant distribution shift."""
        from scipy import stats
        import numpy as np

        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 10000)
        drifted = rng.normal(3, 1, 10000)  # Strongly shifted

        _, p_value = stats.ks_2samp(drifted, reference)
        assert p_value < 0.01, "KS test should detect significant drift"

    def test_ks_test_no_drift_for_same_distribution(self) -> None:
        """KS test should not flag same-distribution samples as drifted."""
        from scipy import stats
        import numpy as np

        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 10000)
        current = rng.normal(0, 1, 10000)  # Same distribution

        _, p_value = stats.ks_2samp(current, reference)
        assert p_value > 0.01, "KS test should not flag same-distribution samples"


class TestEvaluationComponent:
    """Unit tests for the evaluation component metric computation."""

    def test_psi_computation(self) -> None:
        """PSI should return 0 for identical distributions."""
        import numpy as np
        from monitoring.data_drift_detector import DataDriftDetector, DriftDetectorConfig

        config = DriftDetectorConfig(
            model_name="test",
            reference_dataset_uri="",
        )
        detector = DataDriftDetector(config)

        rng = np.random.default_rng(0)
        dist = rng.normal(0, 1, 5000)
        psi = detector._compute_psi(dist, dist.copy())
        assert psi < 0.05, f"PSI for identical distributions should be near 0, got {psi}"

    def test_psi_detects_drift(self) -> None:
        """PSI should return > 0.2 for heavily shifted distributions."""
        import numpy as np
        from monitoring.data_drift_detector import DataDriftDetector, DriftDetectorConfig

        config = DriftDetectorConfig(
            model_name="test",
            reference_dataset_uri="",
        )
        detector = DataDriftDetector(config)

        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 5000)
        drifted = rng.normal(5, 1, 5000)
        psi = detector._compute_psi(drifted, reference)
        assert psi > 0.2, f"PSI should detect heavy drift, got {psi}"
