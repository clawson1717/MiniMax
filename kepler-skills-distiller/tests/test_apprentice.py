"""
Tests for Apprentice Model Scaffold.

Tests use a mock model to avoid loading actual weights in CI.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.apprentice import (
    ApprenticeConfig,
    ApprenticeModel,
    DummyApprenticeModel,
    create_apprentice,
)


class TestApprenticeConfig:
    """Tests for ApprenticeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ApprenticeConfig()

        assert config.model_name == "Qwen/Qwen2-1.5B-Instruct"
        assert config.use_4bit is True
        assert config.use_lora is True
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.max_length == 512

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ApprenticeConfig(
            model_name="meta-llama/Llama-3-1B",
            use_4bit=False,
            lora_r=16,
        )

        assert config.model_name == "meta-llama/Llama-3-1B"
        assert config.use_4bit is False
        assert config.lora_r == 16

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ApprenticeConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "model_name" in d
        assert "use_4bit" in d
        assert d["model_name"] == "Qwen/Qwen2-1.5B-Instruct"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "model_name": "test-model",
            "use_4bit": False,
            "lora_r": 32,
        }
        config = ApprenticeConfig.from_dict(d)

        assert config.model_name == "test-model"
        assert config.use_4bit is False
        assert config.lora_r == 32


class TestDummyApprenticeModel:
    """Tests for DummyApprenticeModel (mock for testing)."""

    def test_initialization(self):
        """Test dummy model initialization."""
        model = DummyApprenticeModel()

        assert model.config is not None
        assert model.model is None

    def test_load(self):
        """Test mock load method."""
        model = DummyApprenticeModel()
        model.load()  # Should not raise
    
        assert model.model is not None  # Now initializes on load

    def test_generate(self):
        """Test mock generation."""
        model = DummyApprenticeModel()
        model.load()

        result = model.generate("Test prompt")

        assert "[MOCK]" in result
        assert "Test prompt" in result

    def test_train_step(self):
        """Test mock training step."""
        model = DummyApprenticeModel()
        model.load()

        metrics = model.train_step(None, None, None)

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_save_load_checkpoint(self):
        """Test checkpoint save/load cycle."""
        model = DummyApprenticeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_checkpoint(tmpdir)
            assert Path(tmpdir).exists()

            # Load
            model2 = DummyApprenticeModel()
            model2.load_checkpoint(tmpdir)  # Should not raise

    def test_parameter_counts(self):
        """Test parameter counting methods."""
        model = DummyApprenticeModel()

        trainable = model.get_trainable_parameters()
        total = model.get_total_parameters()

        assert trainable == 1000
        assert total == 1500000000


class TestApprenticeModel:
    """Tests for ApprenticeModel with mock behavior."""

    def test_initialization_default_config(self):
        """Test model initialization with default config."""
        model = ApprenticeModel()

        assert model.config is not None
        assert model.config.model_name == "Qwen/Qwen2-1.5B-Instruct"

    def test_initialization_custom_config(self):
        """Test model initialization with custom config."""
        config = ApprenticeConfig(model_name="custom-model")
        model = ApprenticeModel(config=config)

        assert model.config.model_name == "custom-model"

    def test_device_auto_detection(self):
        """Test device is set correctly."""
        model = ApprenticeModel()

        # Should be either cuda or cpu
        assert model.device in ["cuda", "cpu"]

    def test_generate_without_load_raises(self):
        """Test that generate raises if model not loaded."""
        model = ApprenticeModel()

        with pytest.raises(RuntimeError, match="not loaded"):
            model.generate("test")

    def test_train_step_without_load_raises(self):
        """Test that train_step raises if model not loaded."""
        model = ApprenticeModel()

        import torch
        with pytest.raises(RuntimeError, match="not loaded"):
            model.train_step(
                torch.zeros(1, 10, dtype=torch.long),
                torch.ones(1, 10, dtype=torch.long),
                torch.zeros(1, 10, dtype=torch.long),
            )


class TestCreateApprentice:
    """Tests for create_apprentice factory function."""

    def test_create_mock_model(self):
        """Test creating a mock model."""
        model = create_apprentice(use_mock=True)

        assert isinstance(model, DummyApprenticeModel)

    def test_create_real_model(self):
        """Test creating a real model (returns correct type)."""
        model = create_apprentice(use_mock=False)

        assert isinstance(model, ApprenticeModel)

    def test_create_with_custom_model_name(self):
        """Test creating with custom model name."""
        model = create_apprentice(
            model_name="meta-llama/Llama-3-1B",
            use_mock=True,
        )

        assert model.config.model_name == "meta-llama/Llama-3-1B"

    def test_create_with_extra_kwargs(self):
        """Test creating with additional config options."""
        model = create_apprentice(
            use_mock=True,
            lora_r=32,
            max_length=1024,
        )

        assert model.config.lora_r == 32
        assert model.config.max_length == 1024


class TestCheckpointRoundTrip:
    """Integration tests for checkpoint save/load."""

    def test_config_persistence(self):
        """Test that config is saved and loaded correctly."""
        config = ApprenticeConfig(
            model_name="test-model",
            lora_r=16,
            max_length=256,
        )
        model = DummyApprenticeModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_checkpoint(tmpdir)

            # Check config file exists
            config_path = Path(tmpdir) / "apprentice_config.json"
            assert config_path.exists()

            # Load config
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            assert loaded_config["model_name"] == "test-model"
            assert loaded_config["lora_r"] == 16
            assert loaded_config["max_length"] == 256

    def test_checkpoint_directory_creation(self):
        """Test that checkpoint creates necessary directories."""
        model = DummyApprenticeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "nested" / "checkpoint"
            model.save_checkpoint(checkpoint_path)

            assert checkpoint_path.exists()
