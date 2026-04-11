"""BentoML service for User Persona model inference."""

from __future__ import annotations

import os
from typing import Any

import bentoml
import numpy as np
import numpy.typing as npt

# ── Model runner ──────────────────────────────────────────────────────────────

user_persona_runner = bentoml.mlflow.get("user-persona:Production").to_runner()

# ── Service definition ────────────────────────────────────────────────────────

svc = bentoml.Service(
    name="user-persona-service",
    runners=[user_persona_runner],
)


# ── Input / output schemas ────────────────────────────────────────────────────

class UserPersonaInput(bentoml.io.JSON.pydantic_model()):  # type: ignore[call-arg]
    """Request schema for the /predict endpoint."""

    user_ids: list[int]
    context_features: dict[str, list[float]] | None = None
    return_embeddings: bool = False


class UserPersonaOutput(bentoml.io.JSON.pydantic_model()):  # type: ignore[call-arg]
    """Response schema for the /predict endpoint."""

    user_ids: list[int]
    persona_vectors: list[list[float]]
    segment_ids: list[int]
    confidence_scores: list[float]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@svc.api(
    input=bentoml.io.JSON(pydantic_model=UserPersonaInput),
    output=bentoml.io.JSON(pydantic_model=UserPersonaOutput),
    route="/predict",
)
async def predict(request: UserPersonaInput) -> UserPersonaOutput:
    """Predict persona vectors and segment IDs for a batch of user IDs.

    Args:
        request: Batch of user IDs with optional context features.

    Returns:
        Persona embedding vectors, segment IDs, and per-user confidence scores.
    """
    import pandas as pd

    user_ids = request.user_ids
    features: dict[str, Any] = {"user_id": user_ids}

    if request.context_features:
        features.update(request.context_features)

    input_df = pd.DataFrame(features)

    raw_output = await user_persona_runner.async_run(input_df)

    # Model outputs: [persona_vector (embedding_dim,), segment_id (int), confidence (float)]
    if isinstance(raw_output, np.ndarray):
        embedding_dim = raw_output.shape[1] - 2
        persona_vectors = raw_output[:, :embedding_dim].tolist()
        segment_ids = raw_output[:, embedding_dim].astype(int).tolist()
        confidence_scores = raw_output[:, embedding_dim + 1].tolist()
    else:
        # MLflow pyfunc model returns DataFrame
        persona_vectors = raw_output.iloc[:, :-2].values.tolist()
        segment_ids = raw_output.iloc[:, -2].astype(int).tolist()
        confidence_scores = raw_output.iloc[:, -1].tolist()

    return UserPersonaOutput(
        user_ids=user_ids,
        persona_vectors=persona_vectors,
        segment_ids=segment_ids,
        confidence_scores=confidence_scores,
    )


@svc.api(
    input=bentoml.io.JSON(),
    output=bentoml.io.JSON(),
    route="/health",
)
async def health(_: dict) -> dict:
    """Liveness check — returns model metadata."""
    model_ref = bentoml.mlflow.get("user-persona:Production")
    return {
        "status": "ok",
        "model_name": "user-persona",
        "model_version": model_ref.tag.version,
        "runner_status": "ready",
    }
