import json
import numpy as np
from typing import Dict, Any, List, Optional
from src.skill_generator import ExpertSkillGenerator

class KeplerReasoningAgent:
    """
    Physics-guided reasoning agent for scientific discovery.
    Uses inferred physical properties to guide the formulation of hypotheses.
    """

    def __init__(self, model_client: Any, skill_generator: Optional[ExpertSkillGenerator] = None):
        """
        Initialize with a model client for LLM inference.
        
        Args:
            model_client: An object that supports a 'generate' method.
            skill_generator: Optional generator to distill skills from successful runs.
        """
        self.model_client = model_client
        self.skill_generator = skill_generator

    def reason(self, data: Dict[str, Any], physical_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the "Think like a Scientist" reasoning loop.
        
        Args:
            data: Raw or summarized numerical data.
            physical_constraints: Dictionary of inferred physical properties.
            
        Returns:
            A dictionary containing the reasoning trace, hypotheses, and proposed expressions.
        """
        prompt = self._construct_prompt(data, physical_constraints)
        
        # Call the model client to generate the reasoning
        response_text = self.model_client.generate(prompt)
        
        return self._parse_response(response_text)

    def _construct_prompt(self, data: Dict[str, Any], physical_constraints: Dict[str, Any]) -> str:
        """
        Constructs the scientific reasoning prompt.
        """
        constraints_str = json.dumps(physical_constraints, indent=2)
        
        # Summarize data if it's too large
        data_summary = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data_summary[k] = {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v))
                }
            else:
                data_summary[k] = v
        
        prompt = f"""
You are a Scientific Discovery Agent. Your goal is to find the governing equation for the given data.

### 1. Physical Constraints Analysis
The following physical properties have been inferred from the data:
{constraints_str}

### 2. Data Summary
{json.dumps(data_summary, indent=2)}

### Task: Think like a Scientist
Please follow these steps:
1. Analyze the physical constraints provided. What do they tell you about the system (e.g., conservation laws, symmetries)?
2. Formulate at least three hypotheses for the governing equation based on these constraints and data.
3. Propose specific symbolic expressions to test.

Return your response in JSON format with the following keys:
- "analysis": Your analysis of the constraints.
- "hypotheses": A list of scientific hypotheses.
- "proposed_expressions": A list of symbolic expressions (strings).
"""
        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the model response. 
        Attempts to extract JSON from the response.
        """
        text = response_text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback if the model didn't return valid JSON
            return {
                "analysis": "Failed to parse structured response.",
                "hypotheses": [],
                "proposed_expressions": [],
                "raw_response": response_text
            }

    def distill_skill(self, trajectory: Dict[str, Any], metrics: Dict[str, Any], skill_name: str):
        """
        Triggers skill generation if a generator is configured.
        """
        if self.skill_generator:
            return self.skill_generator.generate_skill(trajectory, metrics, skill_name)
        return None

    def discover(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full discovery pipeline on data.
        
        Args:
            data: Raw numerical data.
            
        Returns:
            Dictionary with discovery results.
        """
        # Use the reason method with empty constraints for basic discovery
        result = self.reason(data, {})
        
        # Add success flag based on whether we found expressions
        result["success"] = len(result.get("proposed_expressions", [])) > 0
        
        # Set best expression
        expressions = result.get("proposed_expressions", [])
        result["best_expression"] = expressions[0] if expressions else None
        
        return result
