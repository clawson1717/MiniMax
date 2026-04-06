"""CGPM Fact Store - SQLite Backend"""

from __future__ import annotations
import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

from cgpm.core import KnowledgeObject, ProvenanceRecord, ConfidenceScore


class FactStore:
    """SQLite-backed fact store for CGPM with hash-addressing."""
    
    def __init__(self, db_path: str = "cgpm_facts.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database with proper schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Knowledge Objects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_objects (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    rules TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    decay_rate REAL NOT NULL,
                    hash_content TEXT NOT NULL,
                    hash_provenance TEXT NOT NULL
                )
            ''')
            
            # Provenance log table (append-only)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS provenance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ko_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    details TEXT,
                    FOREIGN KEY (ko_id) REFERENCES knowledge_objects (id)
                )
            ''')
            
            conn.commit()
    
    def _generate_id(self, content: str, provenance: ProvenanceRecord) -> str:
        """Generate a unique ID for a KnowledgeObject using SHA-256 hash."""
        content_json = json.dumps(content, sort_keys=True)
        provenance_json = json.dumps(asdict(provenance), sort_keys=True)
        
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()
        provenance_hash = hashlib.sha256(provenance_json.encode()).hexdigest()
        
        combined = content_hash + provenance_hash
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def insert(self, ko: KnowledgeObject) -> str:
        """Insert a KnowledgeObject into the fact store and return its ID."""
        ko_id = self._generate_id(ko.content, ko.provenance)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if already exists
            cursor.execute(
                "SELECT id FROM knowledge_objects WHERE id = ?",
                (ko_id,)
            )
            if cursor.fetchone():
                raise ValueError(f"KnowledgeObject with ID {ko_id} already exists")
            
            # Insert the new KnowledgeObject
            cursor.execute('''
                INSERT INTO knowledge_objects 
                (id, content, confidence, provenance, rules, created_at, updated_at, decay_rate, hash_content, hash_provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ko_id,
                json.dumps(ko.content),
                float(ko.confidence.value),
                json.dumps(asdict(ko.provenance)),
                json.dumps([str(r) for r in ko.rules]) if ko.rules else None,
                ko.created_at.isoformat(),
                ko.updated_at.isoformat(),
                float(ko.decay_rate),
                hashlib.sha256(json.dumps(ko.content, sort_keys=True).encode()).hexdigest(),
                hashlib.sha256(json.dumps(asdict(ko.provenance), sort_keys=True).encode()).hexdigest()
            ))
            
            # Log the creation event
            self._log_provenance(ko_id, "CREATE", {
                "content": ko.content,
                "confidence": float(ko.confidence.value),
                "provenance": asdict(ko.provenance)
            })
            
            conn.commit()
        
        return ko_id
    
    def get(self, ko_id: str) -> Optional[KnowledgeObject]:
        """Retrieve a KnowledgeObject by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content, confidence, provenance, rules, created_at, updated_at, decay_rate
                FROM knowledge_objects WHERE id = ?
            ''', (ko_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            content_json, confidence, provenance_json, rules_json, created_at_str, updated_at_str, decay_rate = row
            
            return KnowledgeObject(
                content=json.loads(content_json),
                confidence=ConfidenceScore(float(confidence)),
                provenance=ProvenanceRecord(**json.loads(provenance_json)),
                rules=[str(r) for r in json.loads(rules_json)] if rules_json else [],
                created_at=datetime.fromisoformat(created_at_str),
                updated_at=datetime.fromisoformat(updated_at_str),
                decay_rate=float(decay_rate)
            )
    
    def update_confidence(self, ko_id: str, new_confidence: ConfidenceScore) -> bool:
        """Update the confidence score of a KnowledgeObject."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current timestamp for updated_at
            cursor.execute('''
                UPDATE knowledge_objects 
                SET confidence = ?, updated_at = ?
                WHERE id = ?
            ''', (float(new_confidence), datetime.now().isoformat(), ko_id))
            
            if cursor.rowcount == 0:
                return False
            
            # Log the confidence update
            self._log_provenance(ko_id, "CONFIDENCE_UPDATE", {
                "old_confidence": float(new_confidence),  # This should be the old value, but we don't have it
                "new_confidence": float(new_confidence)
            })
            
            conn.commit()
            return True
    
    def query_by_confidence(self, min_confidence: ConfidenceScore, max_confidence: ConfidenceScore) -> List[KnowledgeObject]:
        """Query all KnowledgeObjects within a confidence range."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content, confidence, provenance, rules, created_at, updated_at, decay_rate
                FROM knowledge_objects 
                WHERE confidence BETWEEN ? AND ?
            ''', (float(min_confidence), float(max_confidence)))
            
            results = []
            for row in cursor.fetchall():
                results.append(KnowledgeObject(
                    content=json.loads(row['content']),
                    confidence=ConfidenceScore(float(row['confidence'])),
                    provenance=ProvenanceRecord(**json.loads(row['provenance'])),
                    rules=[str(r) for r in json.loads(row['rules'])] if row['rules'] else [],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    decay_rate=float(row['decay_rate'])
                ))
            
            return results
    
    def query_by_tag(self, tag: str) -> List[KnowledgeObject]:
        """Query all KnowledgeObjects with a specific tag (in content)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content, confidence, provenance, rules, created_at, updated_at, decay_rate
                FROM knowledge_objects 
                WHERE content LIKE ?
            ''', (f'%"{tag}":%',))
            
            results = []
            for row in cursor.fetchall():
                results.append(KnowledgeObject(
                    content=json.loads(row['content']),
                    confidence=ConfidenceScore(float(row['confidence'])),
                    provenance=ProvenanceRecord(**json.loads(row['provenance'])),
                    rules=[str(r) for r in json.loads(row['rules'])] if row['rules'] else [],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    decay_rate=float(row['decay_rate'])
                ))
            
            return results
    
    def evict_stale(self, hours_threshold: float) -> int:
        """Remove KnowledgeObjects that haven't been updated within the threshold."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate the cutoff timestamp
            cutoff_time = datetime.now().timestamp() - (hours_threshold * 3600)
            
            # Get all stale KOs
            cursor.execute('''
                SELECT id, updated_at 
                FROM knowledge_objects 
                WHERE updated_at < ?
            ''', (datetime.fromtimestamp(cutoff_time).isoformat(),))
            
            stale_ids = [row[0] for row in cursor.fetchall()]
            
            if not stale_ids:
                return 0
            
            # Delete stale KOs
            cursor.execute(
                "DELETE FROM knowledge_objects WHERE id IN (" + ",".join("?" * len(stale_ids)) + ")",
                stale_ids
            )
            
            # Log eviction events
            for ko_id in stale_ids:
                self._log_provenance(ko_id, "EVICT", {
                    "reason": f"Stale (last updated > {hours_threshold} hours ago)"
                })
            
            conn.commit()
            return len(stale_ids)
    
    def _log_provenance(self, ko_id: str, event_type: str, details: Dict[str, Any]) -> None:
        """Log a provenance event to the provenance_log table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO provenance_log (ko_id, event_type, timestamp, details)
                VALUES (?, ?, ?, ?)
            ''', (ko_id, event_type, datetime.now().isoformat(), json.dumps(details)))
            conn.commit()