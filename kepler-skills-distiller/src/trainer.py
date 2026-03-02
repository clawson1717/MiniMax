"""
Apprentice Trainer

Manages the training loop for distilling scientific reasoning skills
from the Kepler Agent into the Apprentice model.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .apprentice import ApprenticeModel, ApprenticeConfig, create_apprentice


logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the Apprentice Trainer."""
    
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_every: int = 500
    eval_every: int = 200
    log_every: int = 10
    
    # Wandb/tensorboard
    use_wandb: bool = False
    use_tensorboard: bool = False
    project_name: str = "kepler-skills-distiller"
    run_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "save_every": self.save_every,
            "eval_every": self.eval_every,
            "log_every": self.log_every,
            "use_wandb": self.use_wandb,
            "use_tensorboard": self.use_tensorboard,
            "project_name": self.project_name,
            "run_name": self.run_name,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


class PedagogicalDataset(Dataset):
    """
    Dataset for pedagogically-structured training data.
    
    Loads data from the synthesizer output (Step 6) and prepares
    it for supervised fine-tuning.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Any,
        max_length: int = 512,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL file with training data.
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from JSONL file."""
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return
        
        with open(self.data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format as instruction-response
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        
        full_text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ApprenticeTrainer:
    """
    Trainer for the Apprentice model.
    
    Handles the training loop with gradient accumulation, checkpointing,
    and optional logging with wandb or tensorboard.
    """
    
    def __init__(
        self,
        model: ApprenticeModel,
        config: Optional[TrainerConfig] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Apprentice model to train.
            config: Trainer configuration.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
        """
        self.model = model
        self.config = config or TrainerConfig()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        
        # Logging
        self.writer = None
        self.wandb_run = None
    
    def setup(self) -> None:
        """Set up optimizer, scheduler, and logging."""
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        # Set up scheduler
        if self.train_dataset is not None:
            num_training_steps = (
                len(self.train_dataset)
                // (self.config.batch_size * self.config.gradient_accumulation_steps)
                * self.config.num_epochs
            )
        else:
            num_training_steps = 1000
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        # Set up logging
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(self.config.output_dir) / "logs"
                self.writer = SummaryWriter(log_dir=str(log_dir))
                logger.info(f"TensorBoard logging to {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")
        
        if self.config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.project_name,
                    name=self.config.run_name,
                    config=self.config.to_dict(),
                )
                logger.info(f"Wandb logging to {self.config.project_name}")
            except ImportError:
                logger.warning("Wandb not available")
    
    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary with final training metrics.
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")
        
        self.setup()
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                
                if not self.model.config.use_4bit:
                    input_ids = input_ids.to(self.model.device)
                    attention_mask = attention_mask.to(self.model.device)
                    labels = labels.to(self.model.device)
                
                # Forward pass
                metrics = self.model.train_step(input_ids, attention_mask, labels)
                loss = torch.tensor(metrics["loss"], requires_grad=True)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        avg_loss = total_loss / num_batches
                        self._log_metrics({
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": self.global_step,
                        })
                    
                    # Checkpointing
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                
                # Evaluation
                if self.eval_dataset and self.global_step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint("final")
        
        return {
            "final_loss": total_loss / num_batches,
            "total_steps": self.global_step,
            "epochs": self.config.num_epochs,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the eval dataset.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        if self.eval_dataset is None:
            return {}
        
        self.model.model.eval()
        
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                
                if not self.model.config.use_4bit:
                    input_ids = input_ids.to(self.model.device)
                    attention_mask = attention_mask.to(self.model.device)
                    labels = labels.to(self.model.device)
                
                metrics = self.model.train_step(input_ids, attention_mask, labels)
                total_loss += metrics["loss"]
                num_batches += 1
        
        self.model.model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"eval_loss": avg_loss}
    
    def save_checkpoint(self, name: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            name: Name for the checkpoint.
        """
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_checkpoint(checkpoint_dir)
        
        # Save trainer state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
        }
        
        state_path = checkpoint_dir / "trainer_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load a training checkpoint.
        
        Args:
            path: Path to checkpoint directory.
        """
        path = Path(path)
        
        # Load model
        self.model.load_checkpoint(path)
        
        # Load trainer state
        state_path = path / "trainer_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            
            if self.optimizer and "optimizer_state" in state:
                self.optimizer.load_state_dict(state["optimizer_state"])
            
            if self.scheduler and "scheduler_state" in state:
                self.scheduler.load_state_dict(state["scheduler_state"])
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def _log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "",
    ) -> None:
        """Log metrics to console and optional backends."""
        # Console logging
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"[{prefix or 'train'}] {metric_str}")
        
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, self.global_step)
        
        # Wandb
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=self.global_step)
    
    def cleanup(self) -> None:
        """Clean up logging resources."""
        if self.writer is not None:
            self.writer.close()
        
        if self.wandb_run is not None:
            import wandb
            wandb.finish()


def create_trainer(
    model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    data_path: Optional[str] = None,
    output_dir: str = "./outputs",
    use_mock: bool = False,
    **trainer_kwargs,
) -> ApprenticeTrainer:
    """
    Factory function to create a complete training setup.
    
    Args:
        model_name: Name of the model to use.
        data_path: Path to training data (JSONL).
        output_dir: Directory for outputs.
        use_mock: If True, use mock model for testing.
        **trainer_kwargs: Additional trainer config options.
        
    Returns:
        Configured ApprenticeTrainer instance.
    """
    # Create model
    model = create_apprentice(model_name=model_name, use_mock=use_mock)
    model.load()
    
    # Create config
    config = TrainerConfig(output_dir=output_dir, **trainer_kwargs)
    
    # Create dataset if path provided
    train_dataset = None
    if data_path and not use_mock:
        train_dataset = PedagogicalDataset(
            data_path=data_path,
            tokenizer=model.tokenizer,
        )
    
    return ApprenticeTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
    )
