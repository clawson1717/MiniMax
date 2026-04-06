"""CGPM Confidence Gate"""

from __future__ import annotations
from typing import List, Optional
from dataclasses import asdict

from cgpm.core import KnowledgeObject, ConfidenceScore


class ConfidenceGate:
    """Gate that blocks KnowledgeObjects below a configurable confidence threshold."""
    
    def __init__(self, default_threshold: ConfidenceScore = ConfidenceScore(0.5)):
        self.default_threshold = default_threshold
        self.custom_thresholds: Dict[str, ConfidenceScore] = {}
    
    def set_threshold(self, ko_id: str, threshold: ConfidenceScore) -> None:
        """Set a custom confidence threshold for a specific KnowledgeObject."""
        self.custom_thresholds[ko_id] = threshold
    
    def clear_threshold(self, ko_id: str) -> None:
        """Clear the custom threshold for a specific KnowledgeObject."""
        if ko_id in self.custom_thresholds:
            del self.custom_thresholds[ko_id]
    
    def is_allowed(self, ko: KnowledgeObject, threshold: Optional[ConfidenceScore] = None) -> bool:
        """
        Check if a KnowledgeObject is allowed through the gate.
        
        Args:
            ko: The KnowledgeObject to check
            threshold: Optional custom threshold; uses default_threshold if None
            
        Returns:
            True if the object's confidence meets or exceeds the threshold, False otherwise
        """
        effective_threshold = threshold if threshold is not None else self.default_threshold
        effective_threshold = self.custom_thresholds.get(ko.ko_id, effective_threshold)
        
        return float(ko.confidence.value) >= float(effective_threshold)
    
    def filter_query(self, kos: List[KnowledgeObject], threshold: Optional[ConfidenceScore] = None) -> List[KnowledgeObject]:
        """
        Filter a list of KnowledgeObjects, returning only those that meet the threshold.
        
        Args:
            kos: List of KnowledgeObjects to filter
            threshold: Optional custom threshold; uses default_threshold if None
            
        Returns:
            List of KnowledgeObjects that pass the threshold
        """
        effective_threshold = threshold if threshold is not None else self.default_threshold
        return [ko for ko in kos if self.is_allowed(ko, effective_threshold)]
    
    def get_reasoning_context(self, kos: List[KnowledgeObject], threshold: Optional[ConfidenceScore] = None) -> str:
        """
        Assemble a reasoning context string containing only facts that pass the threshold.
        
        Args:
            kos: List of KnowledgeObjects to include in the context
            threshold: Optional custom threshold; uses default_threshold if None
            
        Returns:
            String representation of the filtered reasoning context
        """
        allowed = self.filter_query(kos, threshold)
        effective_threshold = threshold if threshold is not None else self.default_threshold
        
        context_lines = [
            f"=== Reasoning Context (Threshold: {float(effective_threshold)}) ==="
        ]
        
        for ko in allowed:
            context_lines.append(f"\n[Confidence: {float(ko.confidence)} / Threshold: {float(effective_threshold)}]")
            context_lines.append(f"ID: {ko.id[:8]}...")
            context_lines.append(f"Content: {ko.content}")
            if ko.rules:
                context_lines.append(f"Rules: {', '.join(ko.rules)}")
            context_lines.append(f"Provenance: {ko.provenance.source} ({ko.provenance.timestamp})")
        
        return "\n".join(context_lines)