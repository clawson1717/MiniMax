import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Any, Optional

class BDRLTrainer:
    """
    PEFT-RL Trainer (LoRA/PPO) for beam-draft-rl.
    Uses LoRA for parameter efficiency and integrates PPO with physics-based rewards.
    """
    def __init__(
        self,
        model_name: str,
        lora_config: Optional[LoraConfig] = None,
        learning_rate: float = 1e-4,
        ppo_epochs: int = 4,
        mini_batch_size: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_range: float = 0.2,
    ):
        self.model_name = model_name
        self.lora_config = lora_config or LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.learning_rate = learning_rate
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = None
        self.optimizer = None

    def setup_model(self):
        """
        Initializes the model with LoRA adapters.
        """
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = get_peft_model(base_model, self.lora_config)
        self.model.print_trainable_parameters()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def collect_trajectories(self, prompts: List[str], max_new_tokens: int = 128) -> Dict[str, torch.Tensor]:
        """
        Collects trajectories (states, actions, logprobs, rewards) by sampling from the model.
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        sequences = outputs.sequences
        # Extract actions (sampled tokens) after the prompt
        prompt_len = inputs.input_ids.shape[1]
        action_ids = sequences[:, prompt_len:]
        
        # Calculate log_probs for the sampled actions
        # Simplified: in a real PPO implementation, we'd need the logprobs from the forward pass
        logits = self.model(sequences).logits[:, prompt_len-1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
        
        # This is a stub for the full trajectory collection
        return {
            "sequences": sequences,
            "action_ids": action_ids,
            "action_log_probs": action_log_probs,
            "attention_mask": sequences.ne(self.tokenizer.pad_token_id).long()
        }

    def compute_loss(
        self,
        old_log_probs: torch.Tensor,
        current_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the PPO clipped objective loss.
        """
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Simplified: we usually add a value loss and entropy bonus
        return policy_loss

    def train_step(self, trajectories: Dict[str, torch.Tensor], rewards: torch.Tensor):
        """
        Performs a single PPO training step using collected trajectories and rewards.
        """
        # In a real PPO, we compute advantages using a Critic model or GAE.
        # Here we use a simplified version for Step 5 implementation scope.
        
        old_log_probs = trajectories["action_log_probs"]
        sequences = trajectories["sequences"]
        prompt_len = sequences.shape[1] - old_log_probs.shape[1]
        
        # Advantage is simple reward for each token in the sequence (naive)
        # or just the sequence-level reward broadcasted
        advantages = rewards.unsqueeze(1).expand_as(old_log_probs)
        
        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass to get current log_probs
            outputs = self.model(sequences)
            logits = outputs.logits[:, prompt_len-1:-1, :]
            current_log_probs = F.log_softmax(logits, dim=-1).gather(
                -1, trajectories["action_ids"].unsqueeze(-1)
            ).squeeze(-1)
            
            loss = self.compute_loss(old_log_probs, current_log_probs, advantages)
            loss.backward()
            self.optimizer.step()

        return loss.item()
