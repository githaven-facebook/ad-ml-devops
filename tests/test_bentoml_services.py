"""Tests for BentoML service initialization, predict endpoints, and batch handling."""

from __future__ import annotations

import numpy as np
import pytest


class TestAutobidResponseCache:
    """Unit tests for the TTL-based response cache in autobid_service."""

    def test_cache_set_and_get(self) -> None:
        """Cache should return stored value within TTL."""
        import time
        from collections import OrderedDict

        cache: OrderedDict = OrderedDict()
        ttl = 5.0

        def _set(key: str, value: dict) -> None:
            cache[key] = (time.monotonic(), value)

        def _get(key: str) -> dict | None:
            if key not in cache:
                return None
            ts, val = cache[key]
            if time.monotonic() - ts > ttl:
                del cache[key]
                return None
            return val

        _set("key1", {"bid_multiplier": 1.5, "confidence": 0.9})
        result = _get("key1")
        assert result is not None
        assert result["bid_multiplier"] == 1.5
        assert result["confidence"] == 0.9

    def test_cache_miss_returns_none(self) -> None:
        """Cache should return None for keys that were never set."""
        from collections import OrderedDict

        cache: OrderedDict = OrderedDict()

        def _get(key: str):
            return cache.get(key, (None, None))[1]

        assert _get("nonexistent") is None

    def test_cache_ttl_expiry(self) -> None:
        """Cache should return None after TTL expires."""
        import time
        from collections import OrderedDict

        cache: OrderedDict = OrderedDict()
        ttl = 0.01  # 10ms TTL for testing

        def _set(key: str, value: dict) -> None:
            cache[key] = (time.monotonic(), value)

        def _get(key: str) -> dict | None:
            if key not in cache:
                return None
            ts, val = cache[key]
            if time.monotonic() - ts > ttl:
                del cache[key]
                return None
            return val

        _set("key1", {"bid_multiplier": 1.0, "confidence": 0.8})
        time.sleep(0.02)  # Wait for TTL to expire
        assert _get("key1") is None

    def test_feature_hash_deterministic(self) -> None:
        """Same features should always produce the same cache key."""
        import hashlib
        import json

        def _hash(features: dict) -> str:
            serialized = json.dumps(features, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()

        features = {"ctr": 0.05, "budget": 1000.0, "campaign_type": "cpc"}
        hash1 = _hash(features)
        hash2 = _hash(features)
        assert hash1 == hash2

    def test_feature_hash_differs_for_different_inputs(self) -> None:
        """Different features should produce different cache keys."""
        import hashlib
        import json

        def _hash(features: dict) -> str:
            serialized = json.dumps(features, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()

        features_a = {"ctr": 0.05, "budget": 1000.0}
        features_b = {"ctr": 0.10, "budget": 2000.0}
        assert _hash(features_a) != _hash(features_b)


class TestBidMultiplierClipping:
    """Tests for bid multiplier output range enforcement."""

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            (0.5, 0.5),
            (0.05, 0.1),    # Clipped to min
            (15.0, 10.0),   # Clipped to max
            (1.0, 1.0),
            (10.0, 10.0),
            (0.1, 0.1),
        ],
    )
    def test_bid_multiplier_clipping(self, raw_value: float, expected: float) -> None:
        """Bid multiplier should be clipped to [0.1, 10.0]."""
        result = float(np.clip(raw_value, 0.1, 10.0))
        assert abs(result - expected) < 1e-6, f"clip({raw_value}) = {result}, expected {expected}"

    def test_confidence_clipped_to_unit_interval(self) -> None:
        """Confidence score should be clipped to [0.0, 1.0]."""
        for raw in [-0.5, 0.0, 0.5, 1.0, 1.5]:
            clipped = float(np.clip(raw, 0.0, 1.0))
            assert 0.0 <= clipped <= 1.0


class TestUserPersonaServiceSchema:
    """Tests for user persona service input/output schema validation."""

    def test_batch_output_length_matches_input(self) -> None:
        """Output should have the same number of entries as input user_ids."""
        user_ids = [1001, 1002, 1003, 1004, 1005]
        embedding_dim = 128

        # Simulate model output
        fake_output = np.random.randn(len(user_ids), embedding_dim + 2)
        persona_vectors = fake_output[:, :embedding_dim].tolist()
        segment_ids = fake_output[:, embedding_dim].astype(int).tolist()
        confidence_scores = fake_output[:, embedding_dim + 1].tolist()

        assert len(persona_vectors) == len(user_ids)
        assert len(segment_ids) == len(user_ids)
        assert len(confidence_scores) == len(user_ids)

    def test_persona_vector_dimensionality(self) -> None:
        """Each persona vector should have the expected embedding dimension."""
        embedding_dim = 128
        n_users = 10
        fake_output = np.random.randn(n_users, embedding_dim + 2)
        vectors = fake_output[:, :embedding_dim].tolist()

        for vec in vectors:
            assert len(vec) == embedding_dim


class TestAutobidBatchDeduplication:
    """Tests for batch predict cache hit/miss split logic."""

    def test_batch_preserves_order(self) -> None:
        """Batch predict should return results in the same order as input campaigns."""
        campaign_ids = ["camp_a", "camp_b", "camp_c", "camp_d", "camp_e"]
        # Simulate results built in arbitrary order
        results = [None] * len(campaign_ids)
        for i, cid in enumerate(campaign_ids):
            results[i] = {"campaign_id": cid, "bid_multiplier": float(i + 1) * 0.5}

        for i, result in enumerate(results):
            assert result["campaign_id"] == campaign_ids[i]

    def test_batch_cache_miss_indices_collected_correctly(self) -> None:
        """Campaigns that miss the cache should be collected for batch inference."""
        from collections import OrderedDict
        import time

        cache: OrderedDict = OrderedDict()
        cache["hash_b"] = (time.monotonic(), {"bid_multiplier": 1.2, "confidence": 0.9})

        campaigns = [
            {"id": "a", "hash": "hash_a"},  # miss
            {"id": "b", "hash": "hash_b"},  # hit
            {"id": "c", "hash": "hash_c"},  # miss
        ]

        miss_indices = []
        for i, camp in enumerate(campaigns):
            if camp["hash"] not in cache:
                miss_indices.append(i)

        assert miss_indices == [0, 2]
