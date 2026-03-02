import os
import shutil
import pytest
from src.evaluation import Evaluator

@pytest.fixture
def evaluator():
    test_results_dir = "test_results"
    if os.path.exists(test_results_dir):
        shutil.rmtree(test_results_dir)
    ev = Evaluator(results_dir=test_results_dir)
    yield ev
    if os.path.exists(test_results_dir):
        shutil.rmtree(test_results_dir)

def test_load_benchmarks(evaluator):
    benchmarks = evaluator.load_benchmarks()
    assert "LegalBench" in benchmarks
    assert "MedicalQA" in benchmarks
    assert len(benchmarks["LegalBench"]) > 0

def test_run_evaluation_metrics(evaluator):
    domain = "TestDomain"
    benchmark_items = [
        {
            "query": "Test query",
            "expert_reference": "Test reference",
            "domain_configs": [{"id": "agent1", "domain": "test"}]
        }
    ]
    
    # Run evaluation with small iterations/runs for speed
    report = evaluator.run_evaluation(
        domain, 
        benchmark_items, 
        max_iterations=2, 
        num_runs=2,
        target_gap_reduction=0.01 # Small target to ensure it's hit in mock
    )
    
    assert report["domain"] == domain
    assert "timestamp" in report
    assert len(report["results"]) == 1
    
    result = report["results"][0]
    assert "avg_accuracy_improvement" in result
    assert "convergence_stability" in result
    assert "avg_sample_efficiency" in result
    assert len(result["runs"]) == 2

def test_report_saving(evaluator):
    domain = "SaveTest"
    evaluator.save_report(domain, {"test": "data"})
    
    files = os.listdir(evaluator.results_dir)
    assert any(f.startswith(domain) and f.endswith(".json") for f in files)
