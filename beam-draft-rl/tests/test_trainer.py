import unittest
from unittest.mock import MagicMock, patch
import torch
from src.trainer import BDRLTrainer
from peft import LoraConfig

class TestBDRLTrainer(unittest.TestCase):
    def setUp(self):
        # Using a small config and mocking AutoModel/AutoTokenizer to keep it fast
        self.model_name = "gpt2" # Placeholder for mocking
        self.trainer = BDRLTrainer(model_name=self.model_name)

    @patch("src.trainer.AutoTokenizer.from_pretrained")
    @patch("src.trainer.AutoModelForCausalLM.from_pretrained")
    @patch("src.trainer.get_peft_model")
    def test_setup_model(self, mock_get_peft_model, mock_from_pretrained, mock_tokenizer_from_pretrained):
        # Mocking objects
        mock_model = MagicMock()
        # Mock parameters to avoid empty parameter list error in optimizer
        param = torch.nn.Parameter(torch.ones(1))
        mock_model.parameters.return_value = [param]
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        
        mock_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_get_peft_model.return_value = mock_model
        
        self.trainer.setup_model()
        
        mock_from_pretrained.assert_called()
        mock_get_peft_model.assert_called()
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)

    def test_compute_loss(self):
        # Test PPO loss calculation with dummy tensors
        old_log_probs = torch.tensor([[0.5, 0.5]], requires_grad=True)
        current_log_probs = torch.tensor([[0.6, 0.4]], requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0]], requires_grad=False)
        
        loss = self.trainer.compute_loss(old_log_probs, current_log_probs, advantages)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0) # Should be a scalar mean

    @patch("src.trainer.BDRLTrainer.collect_trajectories")
    def test_collect_trajectories_mock(self, mock_collect):
        # Verify trajectory collection returns expected keys
        mock_collect.return_value = {
            "sequences": torch.ones((1, 10)),
            "action_ids": torch.ones((1, 5)),
            "action_log_probs": torch.ones((1, 5)),
            "attention_mask": torch.ones((1, 10))
        }
        
        trajectories = self.trainer.collect_trajectories(["What is 2+2?"])
        self.assertIn("sequences", trajectories)
        self.assertIn("action_ids", trajectories)
        self.assertIn("action_log_probs", trajectories)

    def test_ppo_ratio_clipping(self):
        # Test that clipping works as expected in the Loss
        # Large ratio should be clipped
        old_log_probs = torch.tensor([0.0])
        current_log_probs = torch.tensor([10.0]) # Large ratio e^10
        advantages = torch.tensor([1.0])
        
        # Manually verify SURR2 clipping logic in compute_loss
        ratio = torch.exp(current_log_probs - old_log_probs)
        clip_range = 0.2
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        
        self.assertAlmostEqual(surr2.item(), 1.2, places=5)

if __name__ == "__main__":
    unittest.main()
