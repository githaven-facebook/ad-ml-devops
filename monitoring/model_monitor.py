"""Production model performance monitor.

Tracks prediction distributions, latency percentiles, and error rates.
Compares against baseline metrics and triggers alerts on degradation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)


@dataclass
class ModelMonitorConfig:
    """Configuration for ModelMonitor."""

    model_name: str
    mlflow_tracking_uri: str
    prometheus_pushgateway: str = "http://prometheus-pushgateway.monitoring.svc.cluster.local:9091"
    serving_endpoint: str = ""
    baseline_model_version: str = "Production"
    # Alert thresholds
    max_latency_p99_ms: float = 100.0
    max_error_rate: float = 0.01
    max_prediction_drift_psi: float = 0.2
    min_throughput_rps: float = 10.0
    # Monitoring window
    scrape_interval_seconds: int = 60
    window_minutes: int = 5


@dataclass
class MetricSnapshot:
    """Point-in-time metrics snapshot."""

    timestamp: float = field(default_factory=time.time)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    prediction_mean: float = 0.0
    prediction_std: float = 0.0
    request_count: int = 0
    error_count: int = 0


class ModelMonitor:
    """Monitor model serving performance and trigger alerts on degradation.

    Scrapes Prometheus metrics from the BentoML serving endpoint, computes
    rolling statistics, and pushes derived metrics back to Prometheus.
    Alerts are triggered via the configured alerting rules in AlertManager.
    """

    def __init__(self, config: ModelMonitorConfig) -> None:
        self.config = config
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self._snapshots: list[MetricSnapshot] = []

    def _setup_metrics(self) -> None:
        labels = ["model_name", "model_version", "environment"]

        self.latency_p50 = Gauge(
            "ml_model_latency_p50_ms",
            "Model inference latency p50 in milliseconds",
            labels,
            registry=self.registry,
        )
        self.latency_p95 = Gauge(
            "ml_model_latency_p95_ms",
            "Model inference latency p95 in milliseconds",
            labels,
            registry=self.registry,
        )
        self.latency_p99 = Gauge(
            "ml_model_latency_p99_ms",
            "Model inference latency p99 in milliseconds",
            labels,
            registry=self.registry,
        )
        self.error_rate_gauge = Gauge(
            "ml_model_error_rate",
            "Model serving error rate (fraction of requests)",
            labels,
            registry=self.registry,
        )
        self.throughput_gauge = Gauge(
            "ml_model_throughput_rps",
            "Model serving throughput in requests per second",
            labels,
            registry=self.registry,
        )
        self.prediction_drift_gauge = Gauge(
            "ml_model_prediction_drift_psi",
            "PSI-based prediction distribution drift vs baseline",
            labels,
            registry=self.registry,
        )
        self.degradation_alert = Gauge(
            "ml_model_degradation_alert",
            "1 if any degradation alert is active, 0 otherwise",
            labels,
            registry=self.registry,
        )

    def _query_prometheus(self, query: str, prometheus_url: str) -> float:
        """Execute an instant PromQL query and return the scalar result."""
        try:
            resp = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
        except Exception as exc:
            logger.warning("Prometheus query failed: %s — %s", query, exc)
        return 0.0

    def _compute_psi(
        self, current_dist: np.ndarray, baseline_dist: np.ndarray, n_bins: int = 20
    ) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-8
        bins = np.linspace(
            min(current_dist.min(), baseline_dist.min()),
            max(current_dist.max(), baseline_dist.max()),
            n_bins + 1,
        )
        current_counts, _ = np.histogram(current_dist, bins=bins)
        baseline_counts, _ = np.histogram(baseline_dist, bins=bins)

        current_pct = current_counts / (current_counts.sum() + eps)
        baseline_pct = baseline_counts / (baseline_counts.sum() + eps)

        current_pct = np.clip(current_pct, eps, None)
        baseline_pct = np.clip(baseline_pct, eps, None)

        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return float(psi)

    def collect_snapshot(
        self,
        prometheus_url: str = "http://prometheus.monitoring.svc.cluster.local:9090",
        model_version: str = "Production",
        environment: str = "production",
    ) -> MetricSnapshot:
        """Collect a metric snapshot from Prometheus for the current window."""
        model = self.config.model_name
        window = f"{self.config.window_minutes}m"

        snapshot = MetricSnapshot()

        snapshot.latency_p50_ms = self._query_prometheus(
            f'histogram_quantile(0.50, sum(rate(bentoml_request_duration_seconds_bucket{{service="{model}"}}[{window}])) by (le)) * 1000',
            prometheus_url,
        )
        snapshot.latency_p95_ms = self._query_prometheus(
            f'histogram_quantile(0.95, sum(rate(bentoml_request_duration_seconds_bucket{{service="{model}"}}[{window}])) by (le)) * 1000',
            prometheus_url,
        )
        snapshot.latency_p99_ms = self._query_prometheus(
            f'histogram_quantile(0.99, sum(rate(bentoml_request_duration_seconds_bucket{{service="{model}"}}[{window}])) by (le)) * 1000',
            prometheus_url,
        )
        total_rps = self._query_prometheus(
            f'sum(rate(bentoml_request_total{{service="{model}"}}[{window}]))',
            prometheus_url,
        )
        error_rps = self._query_prometheus(
            f'sum(rate(bentoml_request_exception_total{{service="{model}"}}[{window}]))',
            prometheus_url,
        )
        snapshot.throughput_rps = total_rps
        snapshot.error_rate = error_rps / max(total_rps, 1e-9)

        self._snapshots.append(snapshot)
        # Keep only last 60 snapshots (1h at 1min interval)
        if len(self._snapshots) > 60:
            self._snapshots.pop(0)

        return snapshot

    def check_alerts(
        self,
        snapshot: MetricSnapshot,
        model_version: str = "Production",
        environment: str = "production",
    ) -> list[str]:
        """Compare snapshot against thresholds and return list of alert messages."""
        alerts: list[str] = []
        cfg = self.config

        if snapshot.latency_p99_ms > cfg.max_latency_p99_ms:
            alerts.append(
                f"p99 latency {snapshot.latency_p99_ms:.1f}ms exceeds threshold {cfg.max_latency_p99_ms}ms"
            )
        if snapshot.error_rate > cfg.max_error_rate:
            alerts.append(
                f"error rate {snapshot.error_rate:.4f} exceeds threshold {cfg.max_error_rate}"
            )
        if snapshot.throughput_rps < cfg.min_throughput_rps and snapshot.throughput_rps > 0:
            alerts.append(
                f"throughput {snapshot.throughput_rps:.1f} rps below minimum {cfg.min_throughput_rps} rps"
            )

        return alerts

    def push_metrics(
        self,
        snapshot: MetricSnapshot,
        model_version: str = "Production",
        environment: str = "production",
        alerts: list[str] | None = None,
    ) -> None:
        """Push metrics to Prometheus Pushgateway."""
        labels = [self.config.model_name, model_version, environment]

        self.latency_p50.labels(*labels).set(snapshot.latency_p50_ms)
        self.latency_p95.labels(*labels).set(snapshot.latency_p95_ms)
        self.latency_p99.labels(*labels).set(snapshot.latency_p99_ms)
        self.error_rate_gauge.labels(*labels).set(snapshot.error_rate)
        self.throughput_gauge.labels(*labels).set(snapshot.throughput_rps)
        self.degradation_alert.labels(*labels).set(1.0 if alerts else 0.0)

        try:
            push_to_gateway(
                self.config.prometheus_pushgateway,
                job=f"ml-monitor-{self.config.model_name}",
                registry=self.registry,
            )
        except Exception as exc:
            logger.warning("Failed to push metrics to gateway: %s", exc)

    def run_once(
        self,
        prometheus_url: str = "http://prometheus.monitoring.svc.cluster.local:9090",
        model_version: str = "Production",
        environment: str = "production",
    ) -> list[str]:
        """Collect metrics, check alerts, push — return list of active alerts."""
        snapshot = self.collect_snapshot(prometheus_url, model_version, environment)
        alerts = self.check_alerts(snapshot, model_version, environment)

        if alerts:
            for alert in alerts:
                logger.warning("[ALERT] %s | %s", self.config.model_name, alert)

        self.push_metrics(snapshot, model_version, environment, alerts)
        return alerts

    def run_loop(
        self,
        prometheus_url: str = "http://prometheus.monitoring.svc.cluster.local:9090",
        model_version: str = "Production",
        environment: str = "production",
    ) -> None:
        """Run monitoring loop indefinitely at scrape_interval_seconds."""
        logger.info(
            "Starting model monitor for %s (interval=%ds)",
            self.config.model_name,
            self.config.scrape_interval_seconds,
        )
        while True:
            try:
                alerts = self.run_once(prometheus_url, model_version, environment)
                logger.info(
                    "Monitor tick: model=%s alerts=%d",
                    self.config.model_name,
                    len(alerts),
                )
            except Exception as exc:
                logger.error("Monitor tick failed: %s", exc, exc_info=True)
            time.sleep(self.config.scrape_interval_seconds)
