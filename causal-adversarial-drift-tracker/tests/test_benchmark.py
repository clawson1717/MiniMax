import pytest
import os
import sys
from unittest.mock import MagicMock
from src.benchmark import AdversarialBenchmark
from src.payload import ReasoningPayload

def test_benchmark_generation():
    benchmark = AdversarialBenchmark()
    length = 10
    drift_index = 5
    chain, root_intent = benchmark.generate_drifting_chain(length=length, drift_index=drift_index)
    
    assert len(chain) == length
    assert root_intent is not None
    assert chain[drift_index]["is_drift_ground_truth"] is True
    assert chain[drift_index-1]["is_drift_ground_truth"] is False
    assert chain[drift_index]["is_origin"] is True

def test_benchmark_run_test_case():
    benchmark = AdversarialBenchmark()
    # High intensity so we definitely detect it
    res = benchmark.run_test_case(length=8, drift_index=4, drift_intensity=10.0)
    
    assert "metrics" in res
    assert "precision" in res["metrics"]
    assert "recall" in res["metrics"]
    assert "false_positive_rate" in res["metrics"]

def test_benchmark_batch():
    benchmark = AdversarialBenchmark()
    summary = benchmark.batch_benchmark(iterations=3)
    
    assert summary["iterations"] == 3
    assert "avg_precision" in summary
    assert "avg_recall" in summary
    assert "origin_pinpoint_accuracy" in summary

def test_token_savings_logic():
    # Token savings (placeholder) is conceptually:
    # (Total nodes if full restart - Nodes actually processed after drift) / Total nodes if full restart
    # In our current benchmark, we simulate processing each step.
    # In a real system, the savings come from not having to regenerate correctly processed nodes.
    benchmark = AdversarialBenchmark()
    res = benchmark.run_test_case(length=10, drift_index=3, drift_intensity=20.0)
    
    # We should have at least one healing event if drift was detected
    if res["metrics"]["recall"] > 0:
        assert res["metrics"]["healing_events"] >= 1
