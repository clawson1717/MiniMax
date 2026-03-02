import unittest
from unittest.mock import MagicMock
import numpy as np
import json
from src.reasoning_agent import KeplerReasoningAgent
from src.inferencer import PhysicalPropertyInferencer

class TestKeplerReasoningAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.agent = KeplerReasoningAgent(self.mock_model)

    def test_integration_with_inferencer(self):
        # Create data that will produce specific constraints
        t = np.linspace(0, 10, 100)
        x = np.cos(t)
        y = np.sin(t)
        data = {"t": t, "x": x, "y": y}
        
        inferencer = PhysicalPropertyInferencer(data)
        constraints = inferencer.infer_all()
        
        # Mock response
        self.mock_model.generate.return_value = json.dumps({
            "analysis": "Integrated test",
            "hypotheses": [],
            "proposed_expressions": []
        })
        
        # This confirms KeplerReasoningAgent can consume output of PhysicalPropertyInferencer
        result = self.agent.reason(data, constraints)
        self.assertEqual(result["analysis"], "Integrated test")
        
        # Verify prompt received the conservation laws from the inferencer
        call_args = self.mock_model.generate.call_args[0][0]
        self.assertIn("x^2 + y^2", call_args)

    def test_reasoning_loop(self):
        # Sample data and constraints
        data = {
            "t": np.array([0, 1, 2]),
            "x": np.array([1, 0, -1]),
            "y": np.array([0, 1, 0])
        }
        constraints = {
            "conservation_laws": [{"expression": "x^2 + y^2", "type": "sum_of_squares", "value": 1.0}],
            "symmetries": [{"type": "time-translation", "variable": "x", "confidence": "medium"}]
        }

        # Mock response
        expected_response = {
            "analysis": "The system shows conservation of x^2 + y^2, suggesting circular motion.",
            "hypotheses": ["The motion is periodic.", "The governing equation is a circle."],
            "proposed_expressions": ["x**2 + y**2 - 1"]
        }
        self.mock_model.generate.return_value = json.dumps(expected_response)

        # Run reasoning
        result = self.agent.reason(data, constraints)

        # Verify model was called
        self.mock_model.generate.assert_called_once()
        call_args = self.mock_model.generate.call_args[0][0]
        
        # Verify prompt content (formats constraints correctly)
        self.assertIn("x^2 + y^2", call_args)
        self.assertIn("sum_of_squares", call_args)
        self.assertIn("time-translation", call_args)
        self.assertIn("Think like a Scientist", call_args)

        # Verify result structure (returns structured reasoning)
        self.assertEqual(result["analysis"], expected_response["analysis"])
        self.assertEqual(result["hypotheses"], expected_response["hypotheses"])
        self.assertEqual(result["proposed_expressions"], expected_response["proposed_expressions"])

    def test_parse_response_with_markdown(self):
        # Test that it handles markdown JSON blocks
        raw_response = "Here is the result:\n```json\n{\"analysis\": \"markdown_test\", \"hypotheses\": [], \"proposed_expressions\": []}\n```"
        self.mock_model.generate.return_value = raw_response
        
        result = self.agent.reason({}, {})
        self.assertEqual(result["analysis"], "markdown_test")

    def test_invalid_json_fallback(self):
        # Test fallback for invalid JSON
        self.mock_model.generate.return_value = "This is not JSON"
        
        result = self.agent.reason({}, {})
        self.assertIn("raw_response", result)
        self.assertEqual(result["raw_response"], "This is not JSON")

if __name__ == "__main__":
    unittest.main()
