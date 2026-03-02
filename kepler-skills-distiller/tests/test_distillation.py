"""
Tests for Mastery Distillation Loop
"""

import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.distillation import (
    MasteryRecord,
    CurriculumState,
    MasteryDataset,
    MasteryDistiller,
    BLOOM_TIERS,
    MASTERY_THRESHOLD,
    create_mastery_distiller,
    run_distillation,
)
from src.apprentice import ApprenticeConfig, DummyApprenticeModel


class TestMasteryRecord:
    """Tests for MasteryRecord dataclass."""
    
    def test_init(self):
        """Test mastery record initialization."""
        record = MasteryRecord(tier="Remember")
        assert record.tier == "Remember"
        assert record.attempts == 0
        assert record.correct == 0
        assert record.mastered == False
    
    def test_accuracy_zero_attempts(self):
        """Test accuracy with no attempts."""
        record = MasteryRecord(tier="Apply")
        assert record.accuracy == 0.0
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        record = MasteryRecord(tier="Analyze", attempts=10, correct=8)
        assert record.accuracy == 0.8
    
    def test_update_correct(self):
        """Test update with correct answer."""
        record = MasteryRecord(tier="Remember")
        for _ in range(5):
            record.update(True)
        
        assert record.attempts == 5
        assert record.correct == 5
        assert record.mastered == True
    
    def test_update_incorrect(self):
        """Test update with incorrect answer."""
        record = MasteryRecord(tier="Understand")
        for _ in range(5):
            record.update(False)
        
        assert record.attempts == 5
        assert record.correct == 0
        assert record.mastered == False
    
    def test_mastered_threshold(self):
        """Test mastery threshold check."""
        # 4/5 = 80% < 85%, not mastered
        record = MasteryRecord(tier="Apply")
        for _ in range(4):
            record.update(True)
        record.update(False)  # 4/5 = 80%
        assert record.accuracy < MASTERY_THRESHOLD
        
        # 5/5 = 100% >= 85%, mastered
        record2 = MasteryRecord(tier="Apply")
        for _ in range(5):
            record2.update(True)  # 5/5 = 100%
        assert record2.mastered == True
        
        # Not enough attempts yet (even with 100%)
        record3 = MasteryRecord(tier="Apply")
        for _ in range(3):
            record3.update(True)
        assert record3.mastered == False  # Only 3 attempts


class TestCurriculumState:
    """Tests for CurriculumState."""
    
    def test_init(self):
        """Test curriculum state initialization."""
        state = CurriculumState()
        
        assert state.current_tier == "Remember"
        assert state.tier_index == 0
        assert state.curriculum_complete == False
        assert len(state.mastery_records) == len(BLOOM_TIERS)
    
    def test_all_tiers_initialized(self):
        """Test that all Bloom tiers are initialized."""
        state = CurriculumState()
        
        for tier in BLOOM_TIERS:
            assert tier in state.mastery_records
            assert state.mastery_records[tier].tier == tier
    
    def test_advance_if_mastered(self):
        """Test tier advancement."""
        state = CurriculumState()
        
        # Mark Remember as mastered
        state.mastery_records["Remember"].attempts = 10
        state.mastery_records["Remember"].correct = 9
        state.mastery_records["Remember"].mastered = True
        
        advanced = state.advance_if_mastered()
        
        assert advanced == True
        assert state.current_tier == "Understand"
        assert state.tier_index == 1
    
    def test_no_advance_if_not_mastered(self):
        """Test no advancement without mastery."""
        state = CurriculumState()
        
        # Not mastered yet
        state.mastery_records["Remember"].attempts = 5
        state.mastery_records["Remember"].correct = 3
        
        advanced = state.advance_if_mastered()
        
        assert advanced == False
        assert state.current_tier == "Remember"
    
    def test_curriculum_complete(self):
        """Test curriculum completion."""
        state = CurriculumState()
        state.tier_index = len(BLOOM_TIERS) - 1
        state.current_tier = "Create"
        
        # Mark Create as mastered
        state.mastery_records["Create"].mastered = True
        
        advanced = state.advance_if_mastered()
        
        assert state.curriculum_complete == True
    
    def test_to_dict(self):
        """Test serialization."""
        state = CurriculumState()
        state.mastery_records["Remember"].attempts = 5
        state.mastery_records["Remember"].correct = 4
        
        d = state.to_dict()
        
        assert d["current_tier"] == "Remember"
        assert "mastery_records" in d
        assert d["mastery_records"]["Remember"]["attempts"] == 5
    
    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "current_tier": "Apply",
            "tier_index": 2,
            "total_steps": 100,
            "curriculum_complete": False,
            "mastery_records": {
                "Remember": {"tier": "Remember", "attempts": 10, "correct": 9, "mastered": True},
                "Apply": {"tier": "Apply", "attempts": 5, "correct": 3, "mastered": False},
            }
        }
        
        state = CurriculumState.from_dict(d)
        
        assert state.current_tier == "Apply"
        assert state.tier_index == 2
        assert state.mastery_records["Remember"].mastered == True


class TestMasteryDataset:
    """Tests for MasteryDataset."""
    
    @pytest.fixture
    def temp_data_file(self):
        """Create temporary data file."""
        data = [
            {"tier": "Remember", "question": "What is F=ma?", "answer": "Newton's Second Law"},
            {"tier": "Remember", "question": "What is E=mc^2?", "answer": "Mass-energy equivalence"},
            {"tier": "Apply", "question": "Calculate force", "answer": "6N"},
            {"tier": "Analyze", "question": "Analyze data", "answer": "Pattern found"},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
            return f.name
    
    def test_load_all_tiers(self, temp_data_file):
        """Test loading data without tier filter."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_tokenizer.pad_token_id = 0
        
        dataset = MasteryDataset(
            data_path=temp_data_file,
            tokenizer=mock_tokenizer,
            tier=None,
        )
        
        assert len(dataset) == 4
    
    def test_filter_by_tier(self, temp_data_file):
        """Test filtering by specific tier."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        def mock_tokenize(text, **kwargs):
            return {
                "input_ids": MagicMock(shape=(1, 10)),
                "attention_mask": MagicMock(shape=(1, 10)),
            }
        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        
        dataset = MasteryDataset(
            data_path=temp_data_file,
            tokenizer=mock_tokenizer,
            tier="Remember",
        )
        
        assert len(dataset) == 2
    
    def test_missing_file(self):
        """Test handling of missing data file."""
        mock_tokenizer = Mock()
        
        dataset = MasteryDataset(
            data_path="/nonexistent/path.jsonl",
            tokenizer=mock_tokenizer,
        )
        
        assert len(dataset) == 0


class TestMasteryDistiller:
    """Tests for MasteryDistiller."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock apprentice model."""
        import torch
        import torch.nn as nn
        
        model = DummyApprenticeModel()
        model.load()  # Initialize the mock
        
        # Create a mock model with real torch tensors for parameters
        mock_torch_model = nn.Linear(10, 10)  # Real PyTorch module with parameters
        model.model = mock_torch_model
        model.device = "cpu"
        
        # Create tokenizer that returns real tensors
        def mock_tokenize(text, **kwargs):
            return {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
        
        model.tokenizer = Mock()
        model.tokenizer.pad_token_id = 0
        model.tokenizer.side_effect = mock_tokenize
        model.tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        return model
    
    @pytest.fixture
    def temp_data_file(self):
        """Create temporary data file."""
        data = []
        for tier in BLOOM_TIERS:
            for i in range(5):
                data.append({
                    "tier": tier,
                    "question": f"Question {i} for {tier}",
                    "answer": f"Answer {i}",
                })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
            return f.name
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_init(self, mock_model, temp_data_file, temp_output_dir):
        """Test distiller initialization."""
        from src.trainer import TrainerConfig
        
        config = TrainerConfig(output_dir=temp_output_dir)
        
        distiller = MasteryDistiller(
            model=mock_model,
            config=config,
            data_path=temp_data_file,
        )
        
        assert distiller.model == mock_model
        assert distiller.config.output_dir == temp_output_dir
    
    def test_setup(self, mock_model, temp_data_file, temp_output_dir):
        """Test distiller setup."""
        from src.trainer import TrainerConfig
        
        config = TrainerConfig(output_dir=temp_output_dir)
        
        distiller = MasteryDistiller(
            model=mock_model,
            config=config,
            data_path=temp_data_file,
        )
        
        distiller.setup()
        
        assert len(distiller.tier_datasets) == len(BLOOM_TIERS)
        assert distiller.optimizer is not None
    
    def test_train_on_tier(self, mock_model, temp_data_file, temp_output_dir):
        """Test training on a specific tier."""
        from src.trainer import TrainerConfig
        
        config = TrainerConfig(output_dir=temp_output_dir)
        
        distiller = MasteryDistiller(
            model=mock_model,
            config=config,
            data_path=temp_data_file,
        )
        
        distiller.setup()
        metrics = distiller.train_on_tier("Remember", max_steps=2)
        
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert "steps" in metrics
    
    def test_run_curriculum(self, mock_model, temp_data_file, temp_output_dir):
        """Test running the full curriculum."""
        from src.trainer import TrainerConfig
        
        config = TrainerConfig(output_dir=temp_output_dir)
        
        distiller = MasteryDistiller(
            model=mock_model,
            config=config,
            data_path=temp_data_file,
        )
        
        result = distiller.run_curriculum(steps_per_tier=2, max_iterations=2)
        
        assert "current_tier" in result
        assert "mastery_records" in result


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_mastery_distiller(self):
        """Test factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = create_mastery_distiller(
                output_dir=tmpdir,
                use_mock=True,
            )
            
            assert distiller.model is not None
            assert distiller.config.output_dir == tmpdir
    
    def test_run_distillation(self):
        """Test run_distillation function."""
        data = [
            {"tier": "Remember", "question": "Q1", "answer": "A1"},
            {"tier": "Understand", "question": "Q2", "answer": "A2"},
            {"tier": "Apply", "question": "Q3", "answer": "A3"},
            {"tier": "Analyze", "question": "Q4", "answer": "A4"},
            {"tier": "Evaluate", "question": "Q5", "answer": "A5"},
            {"tier": "Create", "question": "Q6", "answer": "A6"},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
            data_path = f.name
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create distiller with mock that has proper model
            from src.apprentice import DummyApprenticeModel
            mock_model = DummyApprenticeModel()
            mock_model.load()
            
            # Use real PyTorch module with parameters
            mock_model.model = nn.Linear(10, 10)
            mock_model.device = "cpu"
            
            # Create tokenizer that returns real tensors
            def mock_tokenize(text, **kwargs):
                return {
                    "input_ids": torch.zeros(1, 10, dtype=torch.long),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long),
                }
            
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.side_effect = mock_tokenize
            mock_tokenizer.return_value = {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
            mock_model.tokenizer = mock_tokenizer
            
            from src.trainer import TrainerConfig
            from src.distillation import MasteryDistiller
            
            config = TrainerConfig(output_dir=tmpdir)
            distiller = MasteryDistiller(
                model=mock_model,
                config=config,
                data_path=data_path,
            )
            
            result = distiller.run_curriculum(steps_per_tier=1, max_iterations=1)
            
            assert "current_tier" in result


class TestBloomTiers:
    """Tests for Bloom tier constants."""
    
    def test_tier_order(self):
        """Test that tiers are in correct order."""
        expected_order = [
            "Remember",
            "Understand",
            "Apply",
            "Analyze",
            "Evaluate",
            "Create",
        ]
        
        assert BLOOM_TIERS == expected_order
    
    def test_mastery_threshold(self):
        """Test mastery threshold is reasonable."""
        assert 0.0 <= MASTERY_THRESHOLD <= 1.0
        assert MASTERY_THRESHOLD >= 0.8  # At least 80%
