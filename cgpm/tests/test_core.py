"""Tests for core CGPM data structures."""

import math
import time

import pytest

from cgpm.core import (
    ConfidenceLevel,
    ConfidenceScore,
    KnowledgeObject,
    ProvenanceRecord,
    RuleConstraint,
)


class TestConfidenceScore:
    def test_init_default(self):
        cs = ConfidenceScore()
        assert cs.value == 0.5
        assert cs.n_observations == 0
        assert cs.reliability == 0.5

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            ConfidenceScore(value=1.5)
        with pytest.raises(ValueError):
            ConfidenceScore(value=-0.1)

    def test_level_high(self):
        cs = ConfidenceScore(value=0.9)
        assert cs.level == ConfidenceLevel.HIGH

    def test_level_medium(self):
        cs = ConfidenceScore(value=0.65)
        assert cs.level == ConfidenceLevel.MEDIUM

    def test_level_low(self):
        cs = ConfidenceScore(value=0.3)
        assert cs.level == ConfidenceLevel.LOW

    def test_bayesian_update(self):
        cs = ConfidenceScore(value=0.8)
        cs.update(0.4, weight=1.0)
        assert 0.5 < cs.value < 0.7  # Should be between the two values
        assert cs.n_observations == 1


class TestProvenanceRecord:
    def test_to_dict(self):
        prov = ProvenanceRecord(
            source="llm_generation",
            agent_id="test-agent",
            context={"model": "gpt-4"},
        )
        d = prov.to_dict()
        assert d["source"] == "llm_generation"
        assert d["agent_id"] == "test-agent"
        assert d["context"]["model"] == "gpt-4"
        assert "timestamp" in d

    def test_roundtrip(self):
        prov = ProvenanceRecord(source="tool_call", tool_name="web_search")
        restored = ProvenanceRecord.from_dict(prov.to_dict())
        assert restored.source == prov.source
        assert restored.tool_name == prov.tool_name


class TestRuleConstraint:
    def test_to_dict(self):
        rule = RuleConstraint(
            rule_type="exclusion",
            predicate="medical_diagnosis_requires_credential",
            domain="medical",
            strength=0.9,
        )
        d = rule.to_dict()
        assert d["rule_type"] == "exclusion"
        assert d["strength"] == 0.9


class TestKnowledgeObject:
    def test_init_basic(self):
        ko = KnowledgeObject(content="The sky is blue.")
        assert ko.content == "The sky is blue."
        assert ko.confidence == 0.5
        assert ko.ko_id.startswith("ko_")
        # "ko_" + 16 (content hash) + 16 (provenance hash) = 35
        assert len(ko.ko_id) == 35

    def test_init_with_confidence(self):
        ko = KnowledgeObject(content="Pi is approximately 3.14.", confidence=0.99)
        assert ko.confidence == 0.99

    def test_init_invalid_confidence(self):
        with pytest.raises(ValueError):
            KnowledgeObject(content="test", confidence=1.5)

    def test_init_with_provenance(self):
        prov = ProvenanceRecord(source="user_input")
        ko = KnowledgeObject(content="Water freezes at 0C.", provenance=prov)
        assert ko.provenance.source == "user_input"

    def test_deterministic_id(self):
        # IDs are based on content + provenance (which includes timestamp).
        # Same content at different times → different IDs (correct).
        # Same content + SAME provenance record → same ID.
        prov = ProvenanceRecord(source="user_input", timestamp=1234567890.0)
        ko1 = KnowledgeObject(content="Same fact", provenance=prov, confidence=0.9)
        ko2 = KnowledgeObject(content="Same fact", provenance=prov, confidence=0.5)
        # Same provenance instance → same ID (confidence not hashed)
        assert ko1.ko_id == ko2.ko_id
        # Different content → different ID
        ko3 = KnowledgeObject(content="Different fact", provenance=prov)
        assert ko3.ko_id != ko1.ko_id

    def test_different_content_different_id(self):
        ko1 = KnowledgeObject(content="Fact A")
        ko2 = KnowledgeObject(content="Fact B")
        assert ko1.ko_id != ko2.ko_id

    def test_decay(self):
        ko = KnowledgeObject(content="Slowly decaying fact", decay_rate=0.1)
        ko.confidence = 1.0
        new_conf = ko.apply_decay(hours_elapsed=10)
        expected = 1.0 * math.exp(-0.1 * 10)
        assert abs(new_conf - expected) < 1e-9
        assert len(ko.decay_history) == 1

    def test_decay_floor(self):
        ko = KnowledgeObject(content="Fact", decay_rate=1.0)
        ko.confidence = 0.01
        ko.apply_decay(hours_elapsed=1000)
        assert ko.confidence == 0.0

    def test_to_dict_roundtrip(self):
        ko = KnowledgeObject(
            content="Roundtrip test",
            confidence=0.85,
            provenance=ProvenanceRecord(source="test"),
            tags=["test", "unit"],
        )
        d = ko.to_dict()
        restored = KnowledgeObject.from_dict(d)
        assert restored.content == ko.content
        assert restored.confidence == ko.confidence
        assert restored.ko_id == ko.ko_id

    def test_to_json(self):
        ko = KnowledgeObject(content="JSON test", confidence=0.7)
        s = ko.to_json()
        restored = KnowledgeObject.from_json(s)
        assert restored.content == ko.content

    def test_tags(self):
        ko = KnowledgeObject(content="Tagged fact", tags=["physics", "verified"])
        assert "physics" in ko.tags
        assert "verified" in ko.tags
