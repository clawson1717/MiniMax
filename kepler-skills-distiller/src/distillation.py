"""
Mastery Distillation Loop

Implements the training loop that distills expert's scientific skills
into the apprentice model using a pedagogical curriculum based on
Bloom's Taxonomy mastery learning.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import time

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .apprentice import ApprenticeModel, ApprenticeConfig, create_apprentice
from .trainer import TrainerConfig, PedagogicalDataset, ApprenticeTrainer


logger = logging.getLogger(__name__)


# Bloom's Taxonomy tiers in order of difficulty
BLOOM_TIERS = [
    "Remember",
    "Understand", 
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

# Mastery threshold (percentage of correct responses to advance)
MASTERY_THRESHOLD = 0.85


@dataclass
class MasteryRecord:
    """Tracks mastery progress for a single tier."""
    tier: str
    attempts: int = 0
    correct: int = 0
    mastered: bool = False
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts > 0 else 0.0
    
    def update(self, is_correct: bool) -> None:
        self.attempts += 1
        if is_correct:
            self.correct += 1
        self.mastered = self.accuracy >= MASTERY_THRESHOLD and self.attempts >= 5


@dataclass
class CurriculumState:
    """Tracks the overall curriculum state."""
    current_tier: str = "Remember"
    tier_index: int = 0
    mastery_records: Dict[str, MasteryRecord] = field(default_factory=dict)
    total_steps: int = 0
    curriculum_complete: bool = False
    
    def __post_init__(self):
        # Initialize records for all tiers
        for tier in BLOOM_TIERS:
            if tier not in self.mastery_records:
                self.mastery_records[tier] = MasteryRecord(tier=tier)
    
    def advance_if_mastered(self) -> bool:
        """Advance to next tier if current is mastered."""
        current_record = self.mastery_records.get(self.current_tier)
        if current_record and current_record.mastered:
            self.tier_index += 1
            if self.tier_index < len(BLOOM_TIERS):
                self.current_tier = BLOOM_TIERS[self.tier_index]
                logger.info(f"Advanced to tier: {self.current_tier}")
                return True
            else:
                self.curriculum_complete = True
                logger.info("Curriculum complete! All tiers mastered.")
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_tier": self.current_tier,
            "tier_index": self.tier_index,
            "mastery_records": {
                k: {"tier": v.tier, "attempts": v.attempts, 
                    "correct": v.correct, "mastered": v.mastered}
                for k, v in self.mastery_records.items()
            },
            "total_steps": self.total_steps,
            "curriculum_complete": self.curriculum_complete,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CurriculumState":
        state = cls()
        state.current_tier = d.get("current_tier", "Remember")
        state.tier_index = d.get("tier_index", 0)
        state.total_steps = d.get("total_steps", 0)
        state.curriculum_complete = d.get("curriculum_complete", False)
        
        records = d.get("mastery_records", {})
        for tier, record_dict in records.items():
            state.mastery_records[tier] = MasteryRecord(
                tier=record_dict["tier"],
                attempts=record_dict["attempts"],
                correct=record_dict["correct"],
                mastered=record_dict["mastered"],
            )
        
        return state


class MasteryDataset(Dataset):
    """
    Dataset that filters pedagogical data by Bloom tier.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Any,
        tier: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize dataset for a specific tier.
        
        Args:
            data_path: Path to JSONL file with training data.
            tokenizer: Tokenizer for encoding text.
            tier: Bloom tier to filter for. If None, include all.
            max_length: Maximum sequence length.
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.tier = tier
        self.max_length = max_length
        self.data = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from JSONL file, optionally filtering by tier."""
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return
        
        with open(self.data_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if self.tier is None or item.get("tier") == self.tier:
                        self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} samples for tier '{self.tier}'")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format as instruction-response
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        prompt = f"### Question ({item.get('tier', 'General')}):\n{question}\n\n### Answer:\n{answer}"
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tier": item.get("tier", "Unknown"),
        }


class MasteryDistiller:
    """
    Implements the mastery-based distillation loop.
    
    Follows Bloom's Mastery Learning: the apprentice must demonstrate
    mastery (85%+ accuracy) at each tier before advancing to the next.
    """
    
    def __init__(
        self,
        model: ApprenticeModel,
        config: Optional[TrainerConfig] = None,
        data_path: Optional[str] = None,
        skills_path: Optional[str] = None,
    ):
        """
        Initialize the mastery distiller.
        
        Args:
            model: Apprentice model to train.
            config: Training configuration.
            data_path: Path to pedagogical training data (JSONL).
            skills_path: Path to expert skills directory.
        """
        self.model = model
        self.config = config or TrainerConfig()
        self.data_path = Path(data_path) if data_path else None
        self.skills_path = Path(skills_path) if skills_path else None
        
        self.curriculum = CurriculumState()
        self.trainer = None
        self.tier_datasets: Dict[str, MasteryDataset] = {}
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
    
    def setup(self) -> None:
        """Set up datasets, optimizer, and training infrastructure."""
        # Load tier-specific datasets
        if self.data_path and self.model.tokenizer:
            for tier in BLOOM_TIERS:
                self.tier_datasets[tier] = MasteryDataset(
                    data_path=self.data_path,
                    tokenizer=self.model.tokenizer,
                    tier=tier,
                    max_length=getattr(self.model.config, 'max_length', 512),
                )
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        logger.info("Mastery Distiller setup complete")
    
    def load_expert_skills(self) -> List[Dict[str, Any]]:
        """Load expert skills from the skills directory."""
        skills = []
        
        if not self.skills_path or not self.skills_path.exists():
            logger.warning(f"Skills directory not found: {self.skills_path}")
            return skills
        
        for skill_file in self.skills_path.glob("*.json"):
            with open(skill_file, "r") as f:
                skill = json.load(f)
                skills.append(skill)
        
        logger.info(f"Loaded {len(skills)} expert skills")
        return skills
    
    def train_on_tier(self, tier: str, max_steps: int = 100) -> Dict[str, float]:
        """
        Train the model on a specific Bloom tier.
        
        Args:
            tier: The Bloom tier to train on.
            max_steps: Maximum training steps for this tier.
            
        Returns:
            Dictionary with training metrics.
        """
        if tier not in self.tier_datasets:
            logger.warning(f"No dataset for tier: {tier}")
            return {"accuracy": 0.0, "loss": 0.0}
        
        dataset = self.tier_datasets[tier]
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for tier: {tier}")
            return {"accuracy": 0.0, "loss": 0.0}
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        self.model.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        steps = 0
        
        for batch in dataloader:
            if steps >= max_steps:
                break
            
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
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (steps + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            steps += 1
            self.global_step += 1
            
            # Track "correctness" (loss below threshold indicates good performance)
            if loss.item() < 1.0:  # Threshold for "correct"
                correct += 1
            total += 1
        
        avg_loss = total_loss / steps if steps > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        # Update mastery record
        mastery = self.curriculum.mastery_records.get(tier)
        if mastery:
            mastery.attempts += total
            mastery.correct += correct
            mastery.mastered = mastery.accuracy >= MASTERY_THRESHOLD and mastery.attempts >= 5
        
        return {"accuracy": accuracy, "loss": avg_loss, "steps": steps}
    
    def evaluate_tier(self, tier: str) -> float:
        """
        Evaluate model performance on a specific tier.
        
        Args:
            tier: The Bloom tier to evaluate.
            
        Returns:
            Accuracy score for the tier.
        """
        if tier not in self.tier_datasets:
            return 0.0
        
        dataset = self.tier_datasets[tier]
        if len(dataset) == 0:
            return 0.0
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        self.model.model.eval()
        correct = 0
        total = 0
        
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
                
                if metrics["loss"] < 1.0:
                    correct += 1
                total += 1
        
        self.model.model.train()
        return correct / total if total > 0 else 0.0
    
    def run_curriculum(
        self,
        steps_per_tier: int = 100,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the full mastery-based curriculum.
        
        Args:
            steps_per_tier: Training steps per tier per iteration.
            max_iterations: Maximum curriculum iterations.
            
        Returns:
            Final curriculum state.
        """
        self.setup()
        
        iteration = 0
        
        while not self.curriculum.curriculum_complete and iteration < max_iterations:
            iteration += 1
            self.curriculum.total_steps += 1
            
            current_tier = self.curriculum.current_tier
            logger.info(f"Iteration {iteration} - Training on tier: {current_tier}")
            
            # Train on current tier
            metrics = self.train_on_tier(current_tier, max_steps=steps_per_tier)
            logger.info(f"Tier {current_tier} - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2%}")
            
            # Check if mastered and advance
            advanced = self.curriculum.advance_if_mastered()
            
            # Log curriculum state
            self._log_curriculum_state()
            
            # Checkpoint
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint-iter{iteration}")
        
        # Final checkpoint
        self.save_checkpoint("curriculum-final")
        
        return self.curriculum.to_dict()
    
    def _log_curriculum_state(self) -> None:
        """Log current curriculum state."""
        state_str = " | ".join(
            f"{tier}: {record.accuracy:.0%}{'✓' if record.mastered else ''}"
            for tier, record in self.curriculum.mastery_records.items()
        )
        logger.info(f"Mastery Progress: {state_str}")
    
    def save_checkpoint(self, name: str) -> None:
        """Save curriculum and model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_checkpoint(checkpoint_dir)
        
        # Save curriculum state
        curriculum_path = checkpoint_dir / "curriculum_state.json"
        with open(curriculum_path, "w") as f:
            json.dump(self.curriculum.to_dict(), f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load curriculum and model checkpoint."""
        path = Path(path)
        
        # Load model
        self.model.load_checkpoint(path)
        
        # Load curriculum state
        curriculum_path = path / "curriculum_state.json"
        if curriculum_path.exists():
            with open(curriculum_path, "r") as f:
                state_dict = json.load(f)
            self.curriculum = CurriculumState.from_dict(state_dict)
        
        logger.info(f"Loaded checkpoint from {path}")


def create_mastery_distiller(
    model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    data_path: Optional[str] = None,
    skills_path: Optional[str] = None,
    output_dir: str = "./outputs",
    use_mock: bool = False,
    **kwargs,
) -> MasteryDistiller:
    """
    Factory function to create a mastery distiller.
    
    Args:
        model_name: Name of the apprentice model.
        data_path: Path to pedagogical training data.
        skills_path: Path to expert skills directory.
        output_dir: Directory for outputs.
        use_mock: If True, use mock model for testing.
        **kwargs: Additional trainer config options.
        
    Returns:
        Configured MasteryDistiller instance.
    """
    model = create_apprentice(model_name=model_name, use_mock=use_mock)
    model.load()
    
    config = TrainerConfig(output_dir=output_dir, **kwargs)
    
    return MasteryDistiller(
        model=model,
        config=config,
        data_path=data_path,
        skills_path=skills_path,
    )


def run_distillation(
    data_path: str,
    output_dir: str = "./outputs",
    steps_per_tier: int = 50,
    use_mock: bool = True,
) -> Dict[str, Any]:
    """
    Run the full distillation pipeline.
    
    Args:
        data_path: Path to pedagogical training data.
        output_dir: Directory for outputs.
        steps_per_tier: Training steps per tier.
        use_mock: Use mock model for testing.
        
    Returns:
        Final curriculum state.
    """
    distiller = create_mastery_distiller(
        data_path=data_path,
        output_dir=output_dir,
        use_mock=use_mock,
    )
    
    return distiller.run_curriculum(steps_per_tier=steps_per_tier)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    result = run_distillation(
        data_path="data/synthetic/synthetic_training.jsonl",
        output_dir="outputs/distillation",
        steps_per_tier=20,
        use_mock=True,
    )
    
    print("\nFinal Curriculum State:")
    print(json.dumps(result, indent=2))
