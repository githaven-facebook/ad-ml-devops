"""Data drift detector using PSI, KL divergence, and JS divergence.

Runs scheduled checks against a reference dataset and sends alerts when
feature distributions shift beyond configured thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectorConfig:
    """Configuration for DataDriftDetector."""

    model_name: str
    reference_dataset_uri: str
    psi_threshold: float = 0.2  # PSI > 0.2 = significant drift
    kl_threshold: float = 0.5  # KL divergence threshold
    js_threshold: float = 0.3  # JS divergence threshold
    n_bins: int = 20
    min_samples: int = 1000
    alert_on_drift: bool = True
    slack_webhook_url: str = ""
    prometheus_pushgateway: str = "http://prometheus-pushgateway.monitoring.svc.cluster.local:9091"


@dataclass
class FeatureDriftResult:
    """Drift analysis result for a single feature."""

    feature_name: str
    psi: float
    kl_divergence: float
    js_divergence: float
    drifted: bool
    drift_severity: str  # "none", "warning", "critical"
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class DriftReport:
    """Full drift report for one analysis run."""

    model_name: str
    timestamp: float
    dataset_uri: str
    reference_uri: str
    features_checked: int
    drifted_features: list[str]
    feature_results: dict[str, FeatureDriftResult]
    overall_drift_detected: bool
    overall_psi: float


class DataDriftDetector:
    """Detect feature distribution drift using PSI, KL, and JS divergence.

    Uses a fixed reference dataset (typically the training data distribution)
    as the baseline. Scheduled checks compare production serving traffic
    samples against this reference.
    """

    def __init__(self, config: DriftDetectorConfig) -> None:
        self.config = config
        self._reference_df: pd.DataFrame | None = None

    def _load_dataframe(self, uri: str) -> pd.DataFrame:
        """Load a parquet dataset from S3/GCS/local path."""
        if uri.startswith("s3://"):
            import boto3

            parts = uri[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(obj["Body"])
        if uri.startswith("gs://"):
            return pd.read_parquet(uri)
        return pd.read_parquet(uri)

    def load_reference(self) -> None:
        """Load (or reload) the reference dataset into memory."""
        logger.info("Loading reference dataset from %s", self.config.reference_dataset_uri)
        self._reference_df = self._load_dataframe(self.config.reference_dataset_uri)
        logger.info(
            "Reference dataset loaded: %d rows, %d columns",
            len(self._reference_df),
            len(self._reference_df.columns),
        )

    @staticmethod
    def _compute_psi(current: np.ndarray, reference: np.ndarray, n_bins: int = 20) -> float:
        """Compute Population Stability Index (PSI).

        PSI < 0.1: no significant drift
        PSI 0.1–0.2: moderate drift (warning)
        PSI > 0.2: significant drift (critical)
        """
        eps = 1e-8
        combined = np.concatenate([current, reference])
        bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

        curr_counts, _ = np.histogram(current, bins=bins)
        ref_counts, _ = np.histogram(reference, bins=bins)

        curr_pct = curr_counts / (curr_counts.sum() + eps)
        ref_pct = ref_counts / (ref_counts.sum() + eps)

        curr_pct = np.clip(curr_pct, eps, None)
        ref_pct = np.clip(ref_pct, eps, None)

        return float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))

    @staticmethod
    def _compute_kl_divergence(
        current: np.ndarray, reference: np.ndarray, n_bins: int = 20
    ) -> float:
        """Compute KL divergence KL(current || reference)."""
        eps = 1e-8
        combined = np.concatenate([current, reference])
        bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

        curr_counts, _ = np.histogram(current, bins=bins)
        ref_counts, _ = np.histogram(reference, bins=bins)

        curr_pct = np.clip(curr_counts / (curr_counts.sum() + eps), eps, None)
        ref_pct = np.clip(ref_counts / (ref_counts.sum() + eps), eps, None)

        return float(np.sum(curr_pct * np.log(curr_pct / ref_pct)))

    @staticmethod
    def _compute_js_divergence(
        current: np.ndarray, reference: np.ndarray, n_bins: int = 20
    ) -> float:
        """Compute Jensen-Shannon divergence (symmetric, bounded [0,1])."""
        eps = 1e-8
        combined = np.concatenate([current, reference])
        bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

        curr_counts, _ = np.histogram(current, bins=bins)
        ref_counts, _ = np.histogram(reference, bins=bins)

        p = np.clip(curr_counts / (curr_counts.sum() + eps), eps, None)
        q = np.clip(ref_counts / (ref_counts.sum() + eps), eps, None)
        m = 0.5 * (p + q)

        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return float(np.clip(js, 0.0, 1.0))

    def _drift_severity(self, psi: float) -> str:
        if psi < 0.1:
            return "none"
        if psi < self.config.psi_threshold:
            return "warning"
        return "critical"

    def analyze_feature(
        self,
        feature_name: str,
        current_values: np.ndarray,
        reference_values: np.ndarray,
    ) -> FeatureDriftResult:
        """Run all drift metrics for a single feature."""
        psi = self._compute_psi(current_values, reference_values, self.config.n_bins)
        kl = self._compute_kl_divergence(current_values, reference_values, self.config.n_bins)
        js = self._compute_js_divergence(current_values, reference_values, self.config.n_bins)

        drifted = (
            psi > self.config.psi_threshold
            or kl > self.config.kl_threshold
            or js > self.config.js_threshold
        )
        severity = self._drift_severity(psi)

        return FeatureDriftResult(
            feature_name=feature_name,
            psi=psi,
            kl_divergence=kl,
            js_divergence=js,
            drifted=drifted,
            drift_severity=severity,
            reference_mean=float(np.mean(reference_values)),
            current_mean=float(np.mean(current_values)),
            reference_std=float(np.std(reference_values)),
            current_std=float(np.std(current_values)),
        )

    def detect(self, current_dataset_uri: str) -> DriftReport:
        """Run full drift detection for all numeric features.

        Args:
            current_dataset_uri: S3/GCS URI for the current production sample.

        Returns:
            DriftReport with per-feature and overall results.
        """
        import time

        if self._reference_df is None:
            self.load_reference()

        assert self._reference_df is not None

        logger.info("Loading current dataset from %s", current_dataset_uri)
        current_df = self._load_dataframe(current_dataset_uri)

        if len(current_df) < self.config.min_samples:
            logger.warning(
                "Current dataset has only %d samples (min %d); drift results may be unreliable",
                len(current_df),
                self.config.min_samples,
            )

        numeric_cols = self._reference_df.select_dtypes(include=["number"]).columns
        shared_cols = [c for c in numeric_cols if c in current_df.columns]

        feature_results: dict[str, FeatureDriftResult] = {}
        drifted_features: list[str] = []

        for col in shared_cols:
            ref_vals = self._reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            result = self.analyze_feature(col, cur_vals, ref_vals)
            feature_results[col] = result

            if result.drifted:
                drifted_features.append(col)
                logger.warning(
                    "Drift detected in feature '%s': PSI=%.3f KL=%.3f JS=%.3f severity=%s",
                    col,
                    result.psi,
                    result.kl_divergence,
                    result.js_divergence,
                    result.drift_severity,
                )

        overall_psi = float(
            np.mean([r.psi for r in feature_results.values()]) if feature_results else 0.0
        )

        report = DriftReport(
            model_name=self.config.model_name,
            timestamp=time.time(),
            dataset_uri=current_dataset_uri,
            reference_uri=self.config.reference_dataset_uri,
            features_checked=len(feature_results),
            drifted_features=drifted_features,
            feature_results=feature_results,
            overall_drift_detected=bool(drifted_features),
            overall_psi=overall_psi,
        )

        logger.info(
            "Drift detection complete: %d/%d features drifted, overall PSI=%.3f",
            len(drifted_features),
            len(feature_results),
            overall_psi,
        )

        if report.overall_drift_detected and self.config.alert_on_drift:
            self._send_drift_alert(report)

        return report

    def _send_drift_alert(self, report: DriftReport) -> None:
        """Send a Slack alert for detected drift."""
        if not self.config.slack_webhook_url:
            return

        try:
            import requests

            drifted = report.drifted_features[:10]
            more = len(report.drifted_features) - 10
            feature_list = ", ".join(f"`{f}`" for f in drifted)
            if more > 0:
                feature_list += f" (+{more} more)"

            payload = {
                "attachments": [
                    {
                        "color": "#e01e5a",
                        "blocks": [
                            {
                                "type": "header",
                                "text": {
                                    "type": "plain_text",
                                    "text": f":warning: Data Drift Detected: {report.model_name}",
                                },
                            },
                            {
                                "type": "section",
                                "fields": [
                                    {"type": "mrkdwn", "text": f"*Model:*\n{report.model_name}"},
                                    {
                                        "type": "mrkdwn",
                                        "text": f"*Overall PSI:*\n{report.overall_psi:.3f}",
                                    },
                                    {
                                        "type": "mrkdwn",
                                        "text": f"*Drifted Features:*\n{len(report.drifted_features)}/{report.features_checked}",
                                    },
                                ],
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Affected Features:*\n{feature_list}",
                                },
                            },
                        ],
                    }
                ]
            }
            resp = requests.post(self.config.slack_webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to send drift alert: %s", exc)
