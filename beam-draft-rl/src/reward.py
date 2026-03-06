import re
from src.engine import PhysicsEngine

class VerifiableReward:
    def __init__(self):
        self.engine = PhysicsEngine()

    def calculate_reward(self, reasoning_output, ground_truth):
        """
        Calculates a reward based on reasoning output.
        Reasoning output is expected to follow a format with equations and a final value.
        """
        # Placeholder for complex parsing logic
        # For now, let's assume ground truth contains final reactions
        
        # Simple extraction of final answer from reasoning
        pattern = r"Final Answer:\s*(.*)"
        match = re.search(pattern, reasoning_output)
        
        if not match:
            return 0.0

        student_answer = match.group(1).strip()
        
        # Binary reward for matching ground truth
        if student_answer == ground_truth:
            return 1.0
        
        # Partial reward could be added here by verifying intermediate steps
        return 0.0

    def verify_equations(self, equations):
        """
        Verifies if the provided list of equations are physically consistent.
        """
        # Placeholder for more complex symbolic verification
        # For now, we'll just check if basic addition/subtraction in the equation strings holds true
        # for a simple case: sum of forces = 0
        
        # Example equation: "5N + 5N - 10N = 0"
        results = []
        for eq in equations:
            try:
                # Basic parsing to see if "expr = 0" is true
                lhs, rhs = eq.split('=')
                lhs_val = self.engine.ureg(lhs.strip()).to('newton').magnitude
                rhs_val = float(rhs.strip())
                results.append(abs(lhs_val - rhs_val) < 1e-6)
            except:
                results.append(False)
        
        return all(results) if results else False
