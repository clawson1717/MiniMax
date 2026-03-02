import re
from typing import Dict, List, Callable, Any

class ChecklistRewardModel:
    """
    Evaluates reasoning traces based on a set of binary criteria.
    Inspired by checklist-based reward models (like CM2) for fine-grained rewards.
    """

    def __init__(self, criteria: Dict[str, Callable[[str], bool]] = None):
        if criteria:
            self.criteria = criteria
        else:
            self.criteria = {
                "contains_thought_header": self._check_thought_header,
                "contains_conclusion": self._check_conclusion,
                "logical_connectors_used": self._check_logical_connectors,
                "step_by_step_format": self._check_step_by_step,
                "not_empty": self._check_not_empty,
                "no_excessive_repetition": self._check_repetition,
                "has_mathematical_expressions": self._check_math,
                "logic_density": self._check_logic_density
            }

    def _check_thought_header(self, trace: str) -> bool:
        return bool(re.search(r"<thought>|Thought:|Reasoning:", trace, re.IGNORECASE))

    def _check_conclusion(self, trace: str) -> bool:
        return bool(re.search(r"Conclusion:|Therefore,|In summary,|Final Answer:|Thus,", trace, re.IGNORECASE))

    def _check_logical_connectors(self, trace: str) -> bool:
        connectors = ["because", "since", "consequently", "however", "moreover", "furthermore", "thus", "therefore", "as a result"]
        return any(connector in trace.lower() for connector in connectors)

    def _check_step_by_step(self, trace: str) -> bool:
        # Check for multiple steps (at least 2)
        steps = re.findall(r"\d+\.|\bStep \d+", trace, re.IGNORECASE)
        return len(steps) >= 2

    def _check_logic_density(self, trace: str) -> bool:
        # Check if there's a good density of logical steps/conclusions relative to length
        words = trace.split()
        if not words:
            return False
        
        logic_indicators = ["therefore", "thus", "consequently", "hence", "so", "because", "since", "if", "then"]
        indicator_count = sum(1 for word in words if word.lower().strip(".,") in logic_indicators)
        
        # Expect at least one indicator per 50 words
        return (indicator_count / len(words)) > 0.02

    def _check_not_empty(self, trace: str) -> bool:
        return len(trace.strip()) > 50  # Enforce some minimum length

    def _check_repetition(self, trace: str) -> bool:
        # Simple check for repeated long strings (basic proxy for looping)
        lines = [line.strip() for line in trace.split("\n") if len(line.strip()) > 20]
        if not lines:
            return True
        unique_lines = set(lines)
        return len(unique_lines) / len(lines) > 0.8 if lines else True

    def _check_math(self, trace: str) -> bool:
        # Check for common math symbols or LaTeX-like syntax
        math_indicators = [r"\+", r"-", r"=", r"\*", r"/", r"\^", r"\\", r"\{", r"\}", r"\d+"]
        return any(re.search(pattern, trace) for pattern in math_indicators)

    def evaluate_trace(self, trace: str) -> Dict[str, bool]:
        """
        Checks a trace against all defined criteria.
        """
        return {name: func(trace) for name, func in self.criteria.items()}

    def calculate_reward(self, trace: str) -> float:
        """
        Returns a scalar reward (0.0 to 1.0) based on percentage of satisfied criteria.
        """
        results = self.evaluate_trace(trace)
        if not results:
            return 0.0
        return sum(results.values()) / len(results)

    def get_reward_breakdown(self, trace: str) -> Dict[str, Any]:
        """
        Returns a breakdown of rewards for debugging.
        """
        results = self.evaluate_trace(trace)
        reward = sum(results.values()) / len(results) if results else 0.0
        return {
            "total_reward": reward,
            "criteria_results": results
        }
