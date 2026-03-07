from typing import List, Dict, Optional, Any

class Corrector:
    """
    Corrector class for generating fine-grained verbal feedback for RL based on DenoiseFlow.
    Acts as part of the Sensing-Regulating-Correcting loop.
    """

    def generate_feedback(self, 
                          trajectory: List[float], 
                          reasoning_output: str, 
                          ground_truth: str) -> str:
        """
        Generate fine-grained verbal feedback based on a trajectory of sensing data, 
        model reasoning, and the final ground truth comparison.

        Args:
            trajectory: Inconsistency trajectory (sensing data over steps).
            reasoning_output: The model's reasoning trace.
            ground_truth: The correct reasoning/answer.

        Returns:
            A string containing feedback for future reinforcement learning.
        """
        if not trajectory:
            return "No sensing data available to provide corrective feedback."
            
        avg_noise = sum(trajectory) / len(trajectory)
        max_noise = max(trajectory)

        feedback_parts = []
        
        # Determine performance relative to ground truth
        is_correct = reasoning_output.strip().lower() == ground_truth.strip().lower()

        if is_correct:
            if avg_noise > 0.5:
                feedback_parts.append("Correct answer, but high uncertainty detected. The reasoning path was noisy and may be fragile.")
            else:
                feedback_parts.append("Correct answer with high confidence and stable reasoning.")
        else:
            feedback_parts.append(f"Incorrect answer. Ground truth: '{ground_truth}'.")
            
            # Use trajectory to pinpoint where it went wrong if possible
            # Identify the step with maximum uncertainty (noise spike)
            noisy_step = trajectory.index(max_noise)
            feedback_parts.append(f"Reasoning failure occurred with a noise spike (score: {max_noise:.2f}) at step {noisy_step}.")
            
            if avg_noise > 0.7:
                feedback_parts.append("Significant noise and inconsistency were detected throughout the entire trajectory.")
            
        return " ".join(feedback_parts)
