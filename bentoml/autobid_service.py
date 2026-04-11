"""BentoML service for Autobidding model inference."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

import bentoml
import numpy as np

# ── Model runner ──────────────────────────────────────────────────────────────

autobid_runner = bentoml.mlflow.get("autobid:Production").to_runner()

# ── In-process response cache (TTL-based, keyed by campaign features hash) ───

_CACHE_TTL_SECONDS = 5.0
_CACHE_MAX_SIZE = 10_000

_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()


def _cache_get(key: str) -> dict | None:
    if key not in _cache:
        return None
    ts, value = _cache[key]
    if time.monotonic() - ts > _CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    _cache.move_to_end(key)
    return value


def _cache_set(key: str, value: dict) -> None:
    if key in _cache:
        _cache.move_to_end(key)
    _cache[key] = (time.monotonic(), value)
    if len(_cache) > _CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def _feature_hash(features: dict) -> str:
    serialized = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


# ── Service definition ────────────────────────────────────────────────────────

svc = bentoml.Service(
    name="autobid-service",
    runners=[autobid_runner],
)


# ── Input / output schemas ────────────────────────────────────────────────────

class AutobidPredictInput(bentoml.io.JSON.pydantic_model()):  # type: ignore[call-arg]
    """Request schema for single-campaign bid prediction."""

    campaign_id: str
    campaign_features: dict[str, float]
    context_features: dict[str, float] | None = None
    use_cache: bool = True


class AutobidPredictOutput(bentoml.io.JSON.pydantic_model()):  # type: ignore[call-arg]
    """Response schema for bid multiplier prediction."""

    campaign_id: str
    bid_multiplier: float
    confidence: float
    cached: bool


class AutobidBatchInput(bentoml.io.JSON.pydantic_model()):  # type: ignore[call-arg]
    """Request schema for batch bid prediction."""

    campaigns: list[AutobidPredictInput]


class AutobidBatchOutput(bentoml.io.JSON.pydantic_model()):  # type: ignore[call-arg]
    """Response schema for batch bid prediction."""

    predictions: list[AutobidPredictOutput]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@svc.api(
    input=bentoml.io.JSON(pydantic_model=AutobidPredictInput),
    output=bentoml.io.JSON(pydantic_model=AutobidPredictOutput),
    route="/predict",
)
async def predict(request: AutobidPredictInput) -> AutobidPredictOutput:
    """Predict bid multiplier for a single campaign.

    Checks the in-process response cache first (TTL=5s). Cache hits skip
    model inference to reduce GPU utilization for high-frequency callers.

    Args:
        request: Campaign features and optional context features.

    Returns:
        Bid multiplier in range [0.1, 10.0] and confidence score.
    """
    import pandas as pd

    all_features: dict[str, Any] = {**request.campaign_features}
    if request.context_features:
        all_features.update(request.context_features)

    cache_key = _feature_hash(all_features)

    if request.use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return AutobidPredictOutput(
                campaign_id=request.campaign_id,
                bid_multiplier=cached["bid_multiplier"],
                confidence=cached["confidence"],
                cached=True,
            )

    input_df = pd.DataFrame([all_features])
    raw_output = await autobid_runner.async_run(input_df)

    if isinstance(raw_output, np.ndarray):
        bid_multiplier = float(np.clip(raw_output[0, 0], 0.1, 10.0))
        confidence = float(np.clip(raw_output[0, 1], 0.0, 1.0)) if raw_output.shape[1] > 1 else 1.0
    else:
        bid_multiplier = float(np.clip(raw_output.iloc[0, 0], 0.1, 10.0))
        confidence = float(np.clip(raw_output.iloc[0, 1], 0.0, 1.0)) if raw_output.shape[1] > 1 else 1.0

    result = {"bid_multiplier": bid_multiplier, "confidence": confidence}
    if request.use_cache:
        _cache_set(cache_key, result)

    return AutobidPredictOutput(
        campaign_id=request.campaign_id,
        bid_multiplier=bid_multiplier,
        confidence=confidence,
        cached=False,
    )


@svc.api(
    input=bentoml.io.JSON(pydantic_model=AutobidBatchInput),
    output=bentoml.io.JSON(pydantic_model=AutobidBatchOutput),
    route="/batch_predict",
)
async def batch_predict(request: AutobidBatchInput) -> AutobidBatchOutput:
    """Predict bid multipliers for a batch of campaigns.

    Separates cache hits from misses, runs inference only on misses, then
    merges results preserving original order.

    Args:
        request: List of campaign prediction requests.

    Returns:
        List of bid multiplier predictions in the same order as input.
    """
    import pandas as pd

    results: list[AutobidPredictOutput | None] = [None] * len(request.campaigns)
    miss_indices: list[int] = []
    miss_features: list[dict] = []

    for i, campaign in enumerate(request.campaigns):
        all_features: dict[str, Any] = {**campaign.campaign_features}
        if campaign.context_features:
            all_features.update(campaign.context_features)

        cache_key = _feature_hash(all_features)
        if campaign.use_cache:
            cached = _cache_get(cache_key)
            if cached is not None:
                results[i] = AutobidPredictOutput(
                    campaign_id=campaign.campaign_id,
                    bid_multiplier=cached["bid_multiplier"],
                    confidence=cached["confidence"],
                    cached=True,
                )
                continue

        miss_indices.append(i)
        miss_features.append(all_features)

    if miss_features:
        batch_df = pd.DataFrame(miss_features)
        raw_output = await autobid_runner.async_run(batch_df)

        for j, original_idx in enumerate(miss_indices):
            campaign = request.campaigns[original_idx]
            if isinstance(raw_output, np.ndarray):
                bid_multiplier = float(np.clip(raw_output[j, 0], 0.1, 10.0))
                confidence = float(np.clip(raw_output[j, 1], 0.0, 1.0)) if raw_output.shape[1] > 1 else 1.0
            else:
                bid_multiplier = float(np.clip(raw_output.iloc[j, 0], 0.1, 10.0))
                confidence = float(np.clip(raw_output.iloc[j, 1], 0.0, 1.0)) if raw_output.shape[1] > 1 else 1.0

            all_features = miss_features[j]
            cache_key = _feature_hash(all_features)
            if campaign.use_cache:
                _cache_set(cache_key, {"bid_multiplier": bid_multiplier, "confidence": confidence})

            results[original_idx] = AutobidPredictOutput(
                campaign_id=campaign.campaign_id,
                bid_multiplier=bid_multiplier,
                confidence=confidence,
                cached=False,
            )

    return AutobidBatchOutput(predictions=[r for r in results if r is not None])


@svc.api(
    input=bentoml.io.JSON(),
    output=bentoml.io.JSON(),
    route="/health",
)
async def health(_: dict) -> dict:
    """Liveness check — returns model metadata and cache stats."""
    model_ref = bentoml.mlflow.get("autobid:Production")
    return {
        "status": "ok",
        "model_name": "autobid",
        "model_version": model_ref.tag.version,
        "runner_status": "ready",
        "cache_size": len(_cache),
        "cache_ttl_seconds": _CACHE_TTL_SECONDS,
    }
