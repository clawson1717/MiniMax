"""CGPM Confidence Engine"""

from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict

from cgpm.core import KnowledgeObject, ProvenanceRecord, ConfidenceScore


class ConfidenceEngine:
    """Engine that computes confidence scores with temporal decay and rule scoring."""
    
    def __init__(self, base_decay_rate: float = 0.01, rule_store=None):
        """
        Initialize the ConfidenceEngine.
        
        Args:
            base_decay_rate: Base decay rate applied to all KnowledgeObjects (0.0-1.0)
            rule_store: Optional rule store for retrieving domain-specific rules
        """
        self.base_decay_rate = base_decay_rate
        self.rule_store = rule_store
    
    def compute_temporal_decay(self, ko: KnowledgeObject, current_time: Optional[datetime] = None) -> float:
        """
        Compute the temporal decay factor for a KnowledgeObject.
        
        Args:
            ko: The KnowledgeObject to compute decay for
            current_time: The current time for decay calculation; uses now if None
            
        Returns:
            Decay factor between 0.0 and 1.0
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Calculate age in hours
        age_hours = (current_time - ko.created_at).total_seconds() / 3600.0
        
        # Apply exponential decay: c(t) = c0 * exp(-decay_rate * t)
        # We return the decay factor (multiplier applied to original confidence)
        decay_factor = math.exp(-ko.decay_rate * age_hours)
        
        return decay_factor
    
    def compute_rule_score(self, ko: KnowledgeObject) -> float:
        """
        Compute a rule-based score for a KnowledgeObject.
        
        Args:
            ko: The KnowledgeObject to compute rule score for
            
        Returns:
            Rule score between 0.0 and 1.0
        """
        if not self.rule_store or not ko.rules:
            return 1.0  # No rules, full confidence
        
        rule_score = 1.0
        for rule_id in ko.rules:
            rule = self.rule_store.get(rule_id) if self.rule_store else None
            if rule:
                # Apply rule-specific scoring
                if rule.get("type") == "exclusion":
                    # Exclusion rule: reduces confidence
                    rule_score *= 0.5
                elif rule.get("type") == "endorsement":
                    # Endorsement rule: increases confidence
                    rule_score *= 1.2
                    if rule_score > 1.0:
                        rule_score = 1.0
        
        return rule_score
    
    def compute_composite_score(self, ko: KnowledgeObject, current_time: Optional[datetime] = None) -> float:
        """
        Compute the composite confidence score considering temporal decay and rules.
        
        Args:
            ko: The KnowledgeObject to compute score for
            current_time: Current time for decay calculation; uses now if None
            
        Returns:
            Composite confidence score between 0.0 and 1.0
        """
        # Get original confidence
        original_confidence = float(ko.confidence.value)
        
        # Compute temporal decay factor
        decay_factor = self.compute_temporal_decay(ko, current_time)
        
        # Compute rule-based multiplier
        rule_score = self.compute_rule_score(ko)
        
        # Compute composite
        composite = original_confidence * decay_factor * rule_score
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, composite))
    
    def update_confidence_with_engine(self, ko: KnowledgeObject, current_time: Optional[datetime] = None) -> ConfidenceScore:
        """
        Update a KnowledgeObject's confidence using the engine's calculations.
        
        Args:
            ko: The KnowledgeObject to update
            current_time: Current time for decay calculation; uses now if None
            
        Returns:
            The new ConfidenceScore
        """
        composite = self.compute_composite_score(ko, current_time)
        ko.confidence = ConfidenceScore(composite)
        return ko.confidence