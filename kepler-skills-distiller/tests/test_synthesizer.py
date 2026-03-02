import os
import json
import pytest
from src.synthesizer import PedagogicalSynthesizer

def test_synthesizer_initialization():
    synthesizer = PedagogicalSynthesizer(output_dir="data/test_synthetic")
    assert synthesizer.output_dir == "data/test_synthetic"
    assert os.path.exists("data/test_synthetic")

def test_generate_data():
    synthesizer = PedagogicalSynthesizer(output_dir="data/test_synthetic")
    num_per_tier = 2
    dataset = synthesizer.synthesize(num_examples_per_tier=num_per_tier)
    
    assert len(dataset) == 6 # 6 Bloom tiers
    for tier in PedagogicalSynthesizer.BLOOM_TIERS:
        assert tier in dataset
        assert len(dataset[tier]) == num_per_tier
        for example in dataset[tier]:
            assert "tier" in example
            assert "question" in example
            assert "answer" in example
            assert example["tier"] == tier

def test_file_persistence():
    output_dir = "data/test_synthetic"
    synthesizer = PedagogicalSynthesizer(output_dir=output_dir)
    synthesizer.synthesize(num_examples_per_tier=1)
    
    assert os.path.exists(os.path.join(output_dir, "bloom_split.json"))
    assert os.path.exists(os.path.join(output_dir, "synthetic_training.jsonl"))
    
    with open(os.path.join(output_dir, "bloom_split.json"), 'r') as f:
        data = json.load(f)
        assert isinstance(data, dict)
        
    with open(os.path.join(output_dir, "synthetic_training.jsonl"), 'r') as f:
        lines = f.readlines()
        assert len(lines) == 6 # 1 example * 6 tiers
