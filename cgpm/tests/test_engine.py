"""CGPM Engine Tests"""

from __future__ import annotations
from datetime import datetime, timedelta

from cgpm.core import KnowledgeObject, ProvenanceRecord, ConfidenceScore
from cgpm.engine import ConfidenceEngine

def test_temporal_decay_calculation():
    """Test the temporal decay calculations."""
    
    engine = ConfidenceEngine(base_decay_rate=0.01)
    
    # Create a KnowledgeObject with high confidence and recent creation
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    ko = KnowledgeObject(
        content="Test fact",
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    # Decay should be close to 1.0 for very recent objects
    decay_factor = engine.compute_temporal_decay(ko)
    assert decay_factor >= 0.9, f"Decay factor should be close to 1.0 for recent objects, got {decay_factor}"
    
    # Create an older object (30 days old)
    old_ko = KnowledgeObject(
        content={"fact": "Old fact"},
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=[],
        created_at=datetime.now() - timedelta(days=30),
        updated_at=datetime.now() - timedelta(days=30),
        decay_rate=0.01
    )
    
    decay_factor_old = engine.compute_temporal_decay(old_ko)
    # Should have decayed significantly
    assert decay_factor_old < 0.7, f"Old object should have decayed more, got {decay_factor_old}"
    
    # Test with custom decay rate
    fast_decay_ko = KnowledgeObject(
        content={"fact": "Fast decay fact"},
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=[],
        created_at=datetime.now() - timedelta(days=30),
        updated_at=datetime.now() - timedelta(days=30),
        decay_rate=0.1  # Fast decay
    )
    
    decay_factor_fast = engine.compute_temporal_decay(fast_decay_ko)
    assert decay_factor_fast < 0.05, f"Fast decay should be very low, got {decay_factor_fast}"


def test_rule_scoring():
    """Test the rule-based scoring."""
    
    # Create a simple rule store mock
    class MockRuleStore:
        def get(self, rule_id: str) -> Optional[Dict[str, Any]]:
            if rule_id == "exclusion_rule":
                return {"type": "exclusion"}
            elif rule_id == "endorsement_rule":
                return {"type": "endorsement"}
            return None
    
    engine = ConfidenceEngine(base_decay_rate=0.01, rule_store=MockRuleStore())
    
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    
    # Test with exclusion rule (should reduce score)
    ko_exclusion = KnowledgeObject(
        content="Fact with exclusion",
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=["exclusion_rule"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    rule_score_exclusion = engine.compute_rule_score(ko_exclusion)
    assert rule_score_exclusion == 0.5, f"Exclusion rule should reduce score to 0.5, got {rule_score_exclusion}"
    
    # Test with endorsement rule (should increase score)
    ko_endorsement = KnowledgeObject(
        content={"fact": "Fact with endorsement"},
        confidence=ConfidenceScore(0.6),
        provenance=provenance,
        rules=["endorsement_rule"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    rule_score_endorsement = engine.compute_rule_score(ko_endorsement)
    assert rule_score_endorsement == 1.2, f"Endorsement rule should increase score to 1.2, got {rule_score_endorsement}"
    
    # Test with both rules (exclusion then endorsement)
    ko_both = KnowledgeObject(
        content={"fact": "Fact with both rules"},
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=["exclusion_rule", "endorsement_rule"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    rule_score_both = engine.compute_rule_score(ko_both)
    # 0.9 * 0.5 = 0.45, * 1.2 = 0.54
    assert abs(rule_score_both - 0.54) < 0.01, f"Both rules should give ~0.54, got {rule_score_both}"
    
    # Test with no rules
    ko_no_rules = KnowledgeObject(
        content={"fact": "Fact with no rules"},
        confidence=ConfidenceScore(0.8),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    rule_score_no_rules = engine.compute_rule_score(ko_no_rules)
    assert rule_score_no_rules == 1.0, "No rules should give score of 1.0"


def test_composite_score_calculation():
    """Test the composite confidence score calculation."""
    
    engine = ConfidenceEngine(base_decay_rate=0.01)
    
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    
    # Recent object with no decay and no rules
    recent_ko = KnowledgeObject(
        content="Recent fact",
        confidence=ConfidenceScore(0.8),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    composite_recent = engine.compute_composite_score(recent_ko)
    assert abs(composite_recent - 0.8) < 0.001, f"Recent object should have composite ~0.8, got {composite_recent}"
    
    # Older object with decay
    old_ko = KnowledgeObject(
        content={"fact": "Old fact"},
        confidence=ConfidenceScore(0.8),
        provenance=provenance,
        rules=[],
        created_at=datetime.now() - timedelta(days=30),
        updated_at=datetime.now() - timedelta(days=30),
        decay_rate=0.01
    )
    
    composite_old = engine.compute_composite_score(old_ko)
    assert composite_old < 0.6, f"Old object should have composite < 0.6, got {composite_old}"
    
    # Object with endorsement rule
    endorsed_ko = KnowledgeObject(
        content={"fact": "Endorsed fact"},
        confidence=ConfidenceScore(0.6),
        provenance=provenance,
        rules=["endorsement_rule"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    # Need rule store for this test
    class MockRuleStore:
        def get(self, rule_id: str) -> Optional[Dict[str, Any]]:
            if rule_id == "endorsement_rule":
                return {"type": "endorsement"}
            return None
    
    engine_with_rules = ConfidenceEngine(base_decay_rate=0.01, rule_store=MockRuleStore())
    composite_endorsed = engine_with_rules.compute_composite_score(endorsed_ko)
    assert composite_endorsed == 0.72, f"Endorsed fact should have composite 0.72, got {composite_endorsed}"


def test_engine_update_confidence():
    """Test updating confidence with the engine."""
    
    engine = ConfidenceEngine(base_decay_rate=0.01)
    
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    
    # Create a recent object
    ko = KnowledgeObject(
        content="Test fact",
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    # Update confidence using engine (should apply decay if object is old enough)
    new_confidence = engine.update_confidence_with_engine(ko)
    assert abs(float(new_confidence) - 0.9) < 0.001, f"Recent object should have similar confidence, got {new_confidence}"
    
    # Create an old object
    old_ko = KnowledgeObject(
        content={"fact": "Old test fact"},
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=[],
        created_at=datetime.now() - timedelta(days=30),
        updated_at=datetime.now() - timedelta(days=30),
        decay_rate=0.01
    )
    
    old_confidence = engine.update_confidence_with_engine(old_ko)
    assert old_confidence < 0.6, f"Old object confidence should be < 0.6 after decay, got {old_confidence}"


if __name__ == "__main__":
    test_temporal_decay_calculation()
    test_rule_scoring()
    test_composite_score_calculation()
    test_engine_update_confidence()
    print("All ConfidenceEngine tests passed!")