import unittest
import pandas as pd
from typing import List, Dict, Any
from src.benchmark import BenchmarkRunner

class MockModel:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response

class TestBenchmark(unittest.TestCase):

    def setUp(self):
        self.dataset = [
            {'id': 1, 'prompt': 'Solve reaction at R0 for beam of length 5m with point load 10N at 2m.', 'target': '6N'},
            {'id': 2, 'prompt': 'Solve reaction at RL for beam of length 5m with point load 10N at 2m.', 'target': '4N'}
        ]

    def test_run_benchmark_baseline(self):
        baseline_model = MockModel("The reaction at R0 is 6N. R0 = 10 * (5-2) / 5 = 6N.")
        runner = BenchmarkRunner()
        results = runner.run_benchmark(baseline_model, self.dataset, model_type="baseline")

        self.assertEqual(len(results), 2)
        self.assertIn('baseline', [r['model_type'] for r in runner.results])
        self.assertGreaterEqual(runner.results[0]['metrics']['task_accuracy'], 0.5)

    def test_run_benchmark_bdrl(self):
        # BDRL uses <|draft_start|> tags
        bdrl_model = MockModel("<|draft_start|> R0 = 10 * (5-2) / 5 = 6N. <|draft_end|> The answer is 6N.")
        runner = BenchmarkRunner()
        results = runner.run_benchmark(bdrl_model, self.dataset, model_type="bdrl")

        self.assertEqual(len(results), 2)
        self.assertIn('bdrl', [r['model_type'] for r in runner.results])
        self.assertGreater(runner.results[0]['metrics']['equation_validity_rate'], 0.0)

    def test_generate_report(self):
        baseline_model = MockModel("Long answer with lots of tokens and some equations: R0 = 6N.")
        bdrl_model = MockModel("<|draft_start|> R0 = 6N <|draft_end|> Final: 6N")
        
        runner = BenchmarkRunner()
        runner.run_benchmark(baseline_model, self.dataset, model_type="baseline")
        runner.run_benchmark(bdrl_model, self.dataset, model_type="bdrl")
        
        report = runner.generate_report()
        self.assertIsInstance(report, pd.DataFrame)
        self.assertEqual(len(report), 2)
        
        # Verify columns exist
        expected_cols = ['model_type', 'avg_token_count', 'equation_validity_rate', 'physical_consistency_rate', 'task_accuracy']
        for col in expected_cols:
            self.assertIn(col, report.columns)
            
        # Verify BDRL typically has fewer tokens than baseline in this mock setup
        baseline_tokens = report[report['model_type'] == 'baseline']['avg_token_count'].values[0]
        bdrl_tokens = report[report['model_type'] == 'bdrl']['avg_token_count'].values[0]
        self.assertLess(bdrl_tokens, baseline_tokens)

if __name__ == '__main__':
    unittest.main()
