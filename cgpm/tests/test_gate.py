"""CGPM Gate Tests"""

from __future__ import annotations
from datetime import datetime

from cgpm.core import KnowledgeObject, ProvenanceRecord, ConfidenceScore
from cgpm.gate import ConfidenceGate

def test_confidence_gate_basic():
    """Test basic gate functionality."""
    
    gate = ConfidenceGate(default_threshold=ConfidenceScore(0.5))
    
    # Create a KnowledgeObject with confidence 0.9
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    ko_high = KnowledgeObject(
        content="High confidence fact",
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=["rule_001"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    # Create a KnowledgeObject with confidence 0.3
    ko_low = KnowledgeObject(
        content="Low confidence fact",
        confidence=ConfidenceScore(0.3),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    # Test is_allowed with default threshold
    assert gate.is_allowed(ko_high) == True, "High confidence object should be allowed"
    assert gate.is_allowed(ko_low) == False, "Low confidence object should be blocked"
    
    # Test filter_query
    results = gate.filter_query([ko_high, ko_low], threshold=ConfidenceScore(0.4))
    assert len(results) == 1, "Should return one object with threshold 0.4"
    assert results[0].id == ko_high.id, "Should return the high confidence object"
    
    # Test filter_query with all blocked
    results = gate.filter_query([ko_low], threshold=ConfidenceScore(0.4))
    assert len(results) == 0, "Should return empty list when all are blocked"


def test_confidence_gate_custom_thresholds():
    """Test custom thresholds for specific KnowledgeObjects."""
    
    gate = ConfidenceGate(default_threshold=ConfidenceScore(0.5))
    
    # Create KnowledgeObjects
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    ko_1 = KnowledgeObject(
        content="Fact 1",
        confidence=ConfidenceScore(0.4),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    ko_2 = KnowledgeObject(
        content="Fact 2",
        confidence=ConfidenceScore(0.4),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    # Set custom threshold for ko_1
    gate.set_threshold(ko_1.id, ConfidenceScore(0.3))
    
    # ko_1 should now pass with custom threshold 0.3
    assert gate.is_allowed(ko_1) == True, "ko_1 should pass with custom threshold 0.3"
    assert gate.is_allowed(ko_2) == False, "ko_2 should still fail with default 0.5"
    
    # Clear custom threshold
    gate.clear_threshold(ko_1.id)
    assert gate.is_allowed(ko_1) == False, "ko_1 should fail after clearing threshold"
    
    # Test that custom thresholds override default
    gate.set_threshold(ko_2.id, ConfidenceScore(0.6))
    # Even though ko_2 has confidence 0.4, the custom threshold 0.6 makes it fail
    assert gate.is_allowed(ko_2) == False, "ko_2 should fail with custom threshold 0.6"


def test_confidence_gate_reasoning_context():
    """Test the reasoning context generation."""
    
    gate = ConfidenceGate(default_threshold=ConfidenceScore(0.5))
    
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    
    # Create several KnowledgeObjects with different confidences
    ko_high = KnowledgeObject(
        content="High confidence fact",
        confidence=ConfidenceScore(0.9),
        provenance=provenance,
        rules=["rule_001"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    ko_low = KnowledgeObject(
        content="Low confidence fact",
        confidence=ConfidenceScore(0.3),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    ko_medium = KnowledgeObject(
        content="Medium confidence fact",
        confidence=ConfidenceScore(0.6),
        provenance=provenance,
        rules=["rule_002"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    kos = [ko_high, ko_low, ko_medium]
    
    # Generate reasoning context with default threshold 0.5
    context = gate.get_reasoning_context(kos)
    
    # Should include high and medium but not low
    assert "High confidence fact" in context, "High confidence fact should be in context"
    assert "Medium confidence fact" in context, "Medium confidence fact should be in context"
    assert "Low confidence fact" not in context, "Low confidence fact should NOT be in context"
    
    # Check threshold display
    assert "Threshold: 0.5" in context, "Threshold should be displayed"
    
    # Check that confidence values are shown
    assert "Confidence: 0.9" in context, "Confidence 0.9 should be shown"
    assert "Confidence: 0.6" in context, "Confidence 0.6 should be shown"


def test_confidence_gate_with_threshold_parameter():
    """Test that passing an explicit threshold overrides the default."""
    
    gate = ConfidenceGate(default_threshold=ConfidenceScore(0.5))
    
    provenance = ProvenanceRecord(
        source="test_source",
        timestamp="2026-04-02T02:21:00"
    )
    ko = KnowledgeObject(
        content="Test fact",
        confidence=ConfidenceScore(0.6),
        provenance=provenance,
        rules=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        decay_rate=0.01
    )
    
    # With explicit threshold 0.7, the object should be blocked
    assert gate.is_allowed(ko, threshold=ConfidenceScore(0.7)) == False, "Should be blocked with threshold 0.7"
    
    # With explicit threshold 0.5, the object should be allowed
    assert gate.is_allowed(ko, threshold=ConfidenceScore(0.5)) == True, "Should be allowed with threshold 0.5"


if __name__ == "__main__":
    test_confidence_gate_basic()
    test_confidence_gate_custom_thresholds()
    test_confidence_gate_reasoning_context()
    test_confidence_gate_with_threshold_parameter()
    print("All ConfidenceGate tests passed!")