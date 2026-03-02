from typing import List, Dict, Any, Optional
import math
from src.uncertainty import UncertaintyEstimator

class FailureMonitor:
    """
    Monitors for failure modes identified in Li et al. (2026).
    Specifically: Reasoning Fatigue and Suggestion Hijacking.
    """
    
    def __init__(self, fatigue_threshold: float = 0.7, hijacking_threshold: float = 0.5):
        self.fatigue_threshold = fatigue_threshold
        self.hijacking_threshold = hijacking_threshold
        self.initial_goal: Optional[str] = None
        
    def set_initial_goal(self, goal: str):
        """Sets the initial goal to track for suggestion hijacking."""
        self.initial_goal = goal

    def calculate_fatigue_score(self, trajectory_graph: Dict[str, Any], current_uncertainty: float) -> float:
        """
        Calculates a reasoning fatigue score based on session length, repetition, and uncertainty.
        """
        nodes = trajectory_graph.get("nodes", [])
        edges = trajectory_graph.get("edges", [])
        
        # 1. Session Length Factor (Logarithmic scaling)
        length_factor = min(1.0, math.log(len(nodes) + 1) / math.log(20 + 1)) if nodes else 0.0
        
        # 2. Repetition Factor
        repetition_factor = 0.0
        if nodes:
            observations = [str(node.get("observation", "")) for node in nodes]
            actions = [str(edge.get("action", "")) for edge in edges]
            
            unique_obs = len(set(observations))
            unique_actions = len(set(actions))
            
            obs_repetition = 1.0 - (unique_obs / len(observations)) if observations else 0.0
            action_repetition = 1.0 - (unique_actions / len(actions)) if actions else 0.0
            repetition_factor = (obs_repetition + action_repetition) / 2.0
        
        # 3. Uncertainty Factor (Direct use of current uncertainty)
        # We also look at the trend of uncertainty if available
        uncertainty_trend = 0.0
        if nodes:
            uncertainties = [node.get("metadata", {}).get("uncertainty", 0.0) for node in nodes]
            if len(uncertainties) > 1:
                # Simple trend: average of last few vs previous
                recent = uncertainties[-3:]
                previous = uncertainties[:-3] if len(uncertainties) > 3 else uncertainties[:1]
                if recent and previous:
                    uncertainty_trend = max(0.0, sum(recent)/len(recent) - sum(previous)/len(previous))
        
        # Weighted average of factors - adjusted weights for better sensitivity
        fatigue_score = (0.3 * length_factor) + (0.4 * repetition_factor) + (0.5 * current_uncertainty) + (0.3 * uncertainty_trend)
        return min(1.0, fatigue_score)

    def detect_suggestion_hijacking(self, last_action: Any, last_observation: Any) -> float:
        """
        Heuristic for detecting when model output deviates sharply from the initial goal.
        """
        if not self.initial_goal:
            return 0.0
            
        hijack_keywords = ["ignore previous", "instead", "disregard", "new instructions"]
        content = str(last_action) + " " + str(last_observation)
        
        keyword_count = sum(1 for word in hijack_keywords if word in content.lower())
        keyword_score = min(1.0, keyword_count / 2.0)
        
        # In a real scenario, we'd use embedding similarity to see if it deviates from the goal.
        # For this implementation, we'll use a placeholder logic.
        return keyword_score

    def check_status(self, 
                     trajectory_graph: Dict[str, Any], 
                     current_uncertainty: Optional[float] = None,
                     votes: Optional[List[Any]] = None,
                     probabilities: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Returns a status report or raises an alert if thresholds are exceeded.
        Can calculate uncertainty from votes or probabilities if not provided.
        """
        if current_uncertainty is None:
            if votes is not None:
                current_uncertainty = UncertaintyEstimator.from_votes(votes)
            elif probabilities is not None:
                current_uncertainty = UncertaintyEstimator.from_probs(probabilities)
            else:
                current_uncertainty = 0.0
                
        fatigue_score = self.calculate_fatigue_score(trajectory_graph, current_uncertainty)
        
        last_node = trajectory_graph["nodes"][-1] if trajectory_graph["nodes"] else {}
        last_edge = trajectory_graph["edges"][-1] if trajectory_graph["edges"] else {}
        
        hijacking_score = self.detect_suggestion_hijacking(
            last_edge.get("action", ""), 
            last_node.get("observation", "")
        )
        
        status = {
            "fatigue_score": fatigue_score,
            "hijacking_score": hijacking_score,
            "alerts": []
        }
        
        if fatigue_score > self.fatigue_threshold:
            status["alerts"].append("Reasoning Fatigue Alert: High score detected.")
            
        if hijacking_score > self.hijacking_threshold:
            status["alerts"].append("Suggestion Hijacking Alert: Deviant pattern detected.")
            
        return status
