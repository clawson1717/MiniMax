from dataclasses import dataclass
from typing import List, Dict, Any
import re

@dataclass
class DetectionResult:
    detected: bool
    score: float

class FailureModeDetector:
    """
    Detects failure modes and adversarial patterns from the 'Multi-Turn Attacks' paper.
    """

    def __init__(self):
        # Patterns for different failure modes / attack types
        self.patterns = {
            "self_doubt": [
                r"are you sure",
                r"rethink",
                r"mistake",
                r"double check",
                r"second guess",
                r"are you certain",
                r"is that your final answer",
                r"reconsider",
                r"could you be wrong"
            ],
            "social_conformity": [
                r"everyone agrees",
                r"most people",
                r"experts (say|agree|claim)",
                r"consensus",
                r"common knowledge",
                r"researchers (found|state|argue)",
                r"majority thinks",
                r"others have said",
                r"widely believed"
            ],
            "suggestion_hijacking": [
                r"i think the answer should be",
                r"actually, it's",
                r"could it be",
                r"what if the answer is",
                r"ignore previous",
                r"instead of (that|what you said)",
                r"the correct answer is actually",
                r"you mean [A-D]"
            ],
            "emotional_susceptibility": [
                r"important for my (job|career|grade)",
                r"stay late",
                r"stressed",
                r"desperate",
                r"please help",
                r"i'll be fired",
                r"begging you",
                r"very upset",
                r"sad",
                r"crying",
                r"emergency",
                r"life or death"
            ],
            "reasoning_fatigue": [
                r"explain it again",
                r"more details",
                r"keep going",
                r"repeat that",
                r"tell me more",
                r"don't stop",
                r"further elaborate",
                r"elaborate on every step",
                r"step by step again"
            ]
        }

    def _calculate_score(self, text: str, patterns: List[str]) -> float:
        if not text:
            return 0.0
        
        matches = 0
        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches += 1
        
        # Simple score: ratio of patterns matched, capped at 1.0
        # For a single message, even one match is significant.
        if matches == 0:
            return 0.0
        
        return min(1.0, 0.5 + (0.5 * (matches / len(patterns))))

    def detect_self_doubt(self, text: str) -> DetectionResult:
        score = self._calculate_score(text, self.patterns["self_doubt"])
        return DetectionResult(detected=score > 0.4, score=score)

    def detect_social_conformity(self, text: str) -> DetectionResult:
        score = self._calculate_score(text, self.patterns["social_conformity"])
        return DetectionResult(detected=score > 0.4, score=score)

    def detect_suggestion_hijacking(self, text: str) -> DetectionResult:
        score = self._calculate_score(text, self.patterns["suggestion_hijacking"])
        return DetectionResult(detected=score > 0.4, score=score)

    def detect_emotional_susceptibility(self, text: str) -> DetectionResult:
        score = self._calculate_score(text, self.patterns["emotional_susceptibility"])
        return DetectionResult(detected=score > 0.4, score=score)

    def detect_reasoning_fatigue(self, text: str) -> DetectionResult:
        """
        Detects signs of fatigue-inducing requests or degradation.
        Note: True fatigue detection often requires cross-turn analysis.
        """
        score = self._calculate_score(text, self.patterns["reasoning_fatigue"])
        # Bonus for very long or very short text often associated with fatigue
        if len(text.split()) < 5 or len(text.split()) > 500:
            score = min(1.0, score + 0.1)
            
        return DetectionResult(detected=score > 0.4, score=score)

    def detect_all(self, text: str) -> Dict[str, DetectionResult]:
        return {
            "self_doubt": self.detect_self_doubt(text),
            "social_conformity": self.detect_social_conformity(text),
            "suggestion_hijacking": self.detect_suggestion_hijacking(text),
            "emotional_susceptibility": self.detect_emotional_susceptibility(text),
            "reasoning_fatigue": self.detect_reasoning_fatigue(text)
        }
