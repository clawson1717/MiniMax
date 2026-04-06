"""CGPM Store Tests"""

import sqlite3
import tempfile
import os
from datetime import datetime
from typing import Any

from cgpm.core import KnowledgeObject, ProvenanceRecord, ConfidenceScore
from cgpm.store import FactStore

def test_fact_store_operations():
    """Test basic CRUD operations on the FactStore."""
    
    # Create a temporary database
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
    try:
        # Initialize store
        store = FactStore(db_path=db_file)
        
        # Create a sample KnowledgeObject
        provenance = ProvenanceRecord(
            source="test_source",
            timestamp="2026-04-02T02:21:00"
        )
        ko = KnowledgeObject(
            content="The capital of France is Paris",
            confidence=ConfidenceScore(0.9),
            provenance=provenance,
            rules=["rule_geo_001"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            decay_rate=0.01
        )
        
        # Test insert
        ko_id = store.insert(ko)
        assert ko_id is not None and len(ko_id) == 64, "Insert should return a valid 64-character ID"
        
        # Test get
        retrieved_ko = store.get(ko_id)
        assert retrieved_ko is not None, "Get should retrieve the inserted object"
        assert retrieved_ko.content == ko.content, "Retrieved content should match inserted content"
        assert float(retrieved_ko.confidence) == 0.9, "Retrieved confidence should match"
        assert retrieved_ko.provenance.source == "test_source", "Retrieved provenance should match"
        
        # Test update_confidence
        new_confidence = ConfidenceScore(0.85)
        success = store.update_confidence(ko_id, new_confidence)
        assert success, "Update confidence should succeed"
        
        updated_ko = store.get(ko_id)
        assert float(updated_ko.confidence) == 0.85, "Updated confidence should be 0.85"
        
        # Test query_by_confidence
        results = store.query_by_confidence(ConfidenceScore(0.8), ConfidenceScore(1.0))
        assert len(results) == 1, "Should find one object in confidence range 0.8-1.0"
        assert results[0].id == ko_id, "The found object should be our KO"
        
        # Test query_by_tag
        results = store.query_by_tag("capitals")
        assert len(results) == 1, "Should find one object with 'capitals' tag"
        assert results[0].content == "The capital of France is Paris"
        
        # Test evict_stale
        # First, insert another object with old updated_at
        old_provenance = ProvenanceRecord(
            source="test_source_old",
            timestamp="2026-01-01T00:00:00"
        )
        old_ko = KnowledgeObject(
            content="Old historical fact",
            confidence=ConfidenceScore(0.7),
            provenance=old_provenance,
            rules=[],
            created_at=datetime.now(),
            updated_at=datetime.now().replace(year=2025),
            decay_rate=0.01
        )
        old_ko_id = store.insert(old_ko)
        
        # Evict stale objects (older than 24 hours)
        evicted_count = store.evict_stale(24)
        assert evicted_count == 1, "Should evict one stale object"
        
        # Verify old object is gone
        assert store.get(old_ko_id) is None, "Old object should be evicted"
        # Verify new object still exists
        assert store.get(ko_id) is not None, "New object should still exist"
        
    finally:
        # Clean up temporary file
        if os.path.exists(db_file):
            os.unlink(db_file)


def test_fact_store_edge_cases():
    """Test edge cases and error conditions."""
    
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
    try:
        store = FactStore(db_path=db_file)
        
        # Test duplicate insertion
        provenance = ProvenanceRecord(
            source="test_source",
            timestamp="2026-04-02T02:21:00"
        )
        ko = KnowledgeObject(
            content="Unique content",
            confidence=ConfidenceScore(0.9),
            provenance=provenance,
            rules=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            decay_rate=0.01
        )
        
        # First insert should succeed
        ko_id = store.insert(ko)
        
        # Second insert with same content+provenance should fail
        try:
            store.insert(ko)
            assert False, "Should raise ValueError for duplicate"
        except ValueError as e:
            assert "already exists" in str(e), "Error should indicate duplicate"
        
        # Test get non-existent
        non_existent = store.get("nonexistent123")
        assert non_existent is None, "Get should return None for non-existent ID"
        
        # Test update_confidence on non-existent
        success = store.update_confidence("nonexistent123", ConfidenceScore(0.5))
        assert not success, "Update should fail for non-existent ID"
        
    finally:
        if os.path.exists(db_file):
            os.unlink(db_file)


def test_fact_store_provenance_logging():
    """Test that provenance events are properly logged."""
    
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
    try:
        store = FactStore(db_path=db_file)
        
        # Create and insert a KnowledgeObject
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
        
        ko_id = store.insert(ko)
        
        # Get provenance count for this KO
        with sqlite3.connect(db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM provenance_log WHERE ko_id = ?', (ko_id,))
            count_row = cursor.fetchone()
            initial_count = count_row['count']
            assert initial_count >= 1, "Should have at least one provenance entry"
        
        # Update confidence
        store.update_confidence(ko_id, ConfidenceScore(0.8))
        
        # Check that another provenance entry was added
        with sqlite3.connect(db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM provenance_log WHERE ko_id = ?', (ko_id,))
            count_row = cursor.fetchone()
            final_count = count_row['count']
            assert final_count == initial_count + 1, "Should have one additional provenance entry"
        
    finally:
        if os.path.exists(db_file):
            os.unlink(db_file)


if __name__ == "__main__":
    test_fact_store_operations()
    test_fact_store_edge_cases()
    test_fact_store_provenance_logging()
    print("All FactStore tests passed!")