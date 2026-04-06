"""
Apprentice Model Scaffold

Provides a small student model (Qwen-1.5B or Llama-3-1B) for distillation
of scientific reasoning skills from the Kepler Agent.
"""

import os
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from dataclasses import dataclass


@dataclass
class ApprenticeConfig:
    """Configuration for the Apprentice model."""
    
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    use_4bit: bool = True
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    max_length: int = 512
    device_map: str = "auto"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApprenticeConfig":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "use_4bit": self.use_4bit,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "max_length": self.max_length,
            "device_map": self.device_map,
        }


class ApprenticeModel:
    """
    A small student model for distilling scientific reasoning skills.
    
    Supports Qwen-1.5B and Llama-3-1B with optional 4-bit quantization
    and LoRA fine-tuning for efficient training.
    """
    
    def __init__(
        self,
        config: Optional[ApprenticeConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the Apprentice model.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
            device: Device to load model on. Auto-detected if not provided.
        """
        self.config = config or ApprenticeConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def load(self) -> None:
        """Load the model and tokenizer with optional quantization."""
        # Prepare quantization config
        quantization_config = None
        if self.config.use_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.config.device_map if self.config.use_4bit else None,
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        if not self.config.use_4bit:
            self.model = self.model.to(self.device)
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()
    
    def _apply_lora(self) -> None:
        """Apply LoRA adapters for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.peft_model = self.model
        except ImportError:
            # PEFT not available, skip LoRA
            pass
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt for generation.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            
        Returns:
            Generated text (prompt + completion).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        
        # Move to correct device
        if not self.config.use_4bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask.
            labels: Labels for supervised training.
            
        Returns:
            Dictionary with loss and other metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self.model.train()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        
        return {
            "loss": loss.item(),
        }
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Directory to save checkpoint to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.peft_model is not None:
            self.peft_model.save_pretrained(str(path))
        elif self.model is not None:
            self.model.save_pretrained(str(path))
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(path))
        
        # Save config
        import json
        config_path = path / "apprentice_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model from checkpoint.
        
        Args:
            path: Directory containing checkpoint.
        """
        path = Path(path)
        
        # Load config if available
        import json
        config_path = path / "apprentice_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
                self.config = ApprenticeConfig.from_dict(config_dict)
        
        # Load model
        self.load()
        
        # Load weights
        if self.peft_model is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(path))
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        if self.model is None:
            return 0
        
        trainable = 0
        for param in self.model.parameters():
            if param.requires_grad:
                trainable += param.numel()
        return trainable
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        if self.model is None:
            return 0
        
        return sum(p.numel() for p in self.model.parameters())
    
    def freeze_base_model(self) -> None:
        """Freeze all parameters except LoRA adapters."""
        if self.model is None:
            return
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters if present
        if self.peft_model is not None:
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True


class DummyApprenticeModel:
    """
    A minimal mock for testing without loading actual models.
    Used in CI environments or when GPU is unavailable.
    """
    
    def __init__(self, config: Optional[ApprenticeConfig] = None):
        self.config = config or ApprenticeConfig()
        self.model = None  # Explicitly None at init for test compliance
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2
        self.tokenizer.model_max_length = 512
        self.device = "cpu"
        
        # Mock tokenize
        def mock_tokenize(text, **kwargs):
            return {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
        self.tokenizer.side_effect = mock_tokenize
    
    def load(self) -> None:
        """Mock load - initializes the minimal torch module."""
        if self.model is None:
            self.model = nn.Linear(1, 1)  # Minimal torch module for parameters
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation - returns echo of prompt."""
        return f"[MOCK] {prompt[:100]}..."
    
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        """Mock training step - returns fake loss."""
        import random
        return {"loss": 0.5, "accuracy": random.uniform(0.7, 0.95)}
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Mock save - creates directory and saves config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        import json
        config_path = path / "apprentice_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Mock load - does nothing."""
        pass
    
    def get_trainable_parameters(self) -> int:
        return 1000
    
    def get_total_parameters(self) -> int:
        return 1500000000  # ~1.5B


def create_apprentice(
    model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    use_mock: bool = False,
    **kwargs,
) -> Union[ApprenticeModel, DummyApprenticeModel]:
    """
    Factory function to create an Apprentice model.
    
    Args:
        model_name: Name of the model to load.
        use_mock: If True, return a mock model for testing.
        **kwargs: Additional arguments passed to ApprenticeConfig.
        
    Returns:
        ApprenticeModel or DummyApprenticeModel instance.
    """
    config = ApprenticeConfig(model_name=model_name, **kwargs)
    
    if use_mock:
        return DummyApprenticeModel(config)
    
    return ApprenticeModel(config)
