import unittest
from unittest.mock import patch, MagicMock
import sys
import torch

# Mock the heavy parts of the agent to avoid requiring GPU/accelerate in CI/tests
with patch('transformers.AutoModelForCausalLM.from_pretrained'), \
     patch('transformers.AutoTokenizer.from_pretrained'), \
     patch('transformers.BitsAndBytesConfig'):
    from src.cli import main

class TestCLI(unittest.TestCase):
    @patch('src.cli.ReasoningAgent')
    @patch('src.cli.MasteryTrainer')
    def test_train_invocation(self, mock_trainer_cls, mock_agent_cls):
        """Test if the train command can be invoked without crashing."""
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.run_epoch.return_value = [
            {"success_score": 0.8, "world_model_capacity": 10.5}
        ]
        
        with patch.object(sys, 'argv', ['cli.py', 'train', '--epochs', '1']):
            try:
                main()
            except SystemExit:
                self.fail("main() raised SystemExit unexpectedly!")
        
        mock_trainer.run_epoch.assert_called()

    @patch('src.cli.ReasoningAgent')
    @patch('src.cli.MasteryTrainer')
    def test_evaluate_invocation(self, mock_trainer_cls, mock_agent_cls):
        """Test if the evaluate command can be invoked without crashing."""
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.evaluate_world_model.return_value = 15.0
        
        with patch.object(sys, 'argv', ['cli.py', 'evaluate']):
            try:
                main()
            except SystemExit:
                self.fail("main() raised SystemExit unexpectedly!")
        
        mock_trainer.evaluate_world_model.assert_called()

    def test_visualize_invocation(self):
        """Test if the visualize command can be invoked without crashing."""
        with patch.object(sys, 'argv', ['cli.py', 'visualize']):
            try:
                main()
            except SystemExit:
                self.fail("main() raised SystemExit unexpectedly!")

if __name__ == '__main__':
    unittest.main()
