import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from src.model_wrapper import DraftWrapper
from src.engine import PhysicsEngine

class BenchmarkRunner:
    """
    Runner for benchmarking RL models against BeamDraft-RL (BDRL).
    """
    def __init__(self, baseline_model: Any = None, bdrl_model: Any = None):
        self.baseline_model = baseline_model
        self.bdrl_model = bdrl_model
        self.results = []
        self.physics_engine = PhysicsEngine()
        self.wrapper = DraftWrapper()

    def run_benchmark(self, model: Any, dataset: List[Dict[str, Any]], model_type: str = "baseline") -> List[Dict[str, Any]]:
        """
        Runs the benchmark for a given model on a dataset.
        """
        trajectories = []
        for sample in dataset:
            # Simulate model inference
            # In a real scenario, this would call model.generate()
            output = model.generate(sample['prompt'])
            
            trajectory = {
                'sample_id': sample.get('id'),
                'prompt': sample['prompt'],
                'output': output,
                'target': sample.get('target'),
                'model_type': model_type
            }
            trajectories.append(trajectory)
        
        metrics = self.compute_metrics(trajectories)
        self.results.append({
            'model_type': model_type,
            'metrics': metrics,
            'trajectories': trajectories
        })
        return trajectories

    def compute_metrics(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes efficiency and accuracy metrics from trajectories.
        """
        total_tokens = 0
        valid_equations = 0
        physically_consistent = 0
        task_correct = 0
        total_count = len(trajectories)

        for traj in trajectories:
            output = traj['output']
            target = traj['target']
            
            # 1. Token count reduction (relative to baseline if applicable)
            total_tokens += len(output.split()) # Simplified token count

            # 2. Equation validity (check for basic structural integrity)
            draft = self.wrapper.extract_draft(output)
            content_to_check = draft if draft else output
            if "=" in content_to_check and any(char.isdigit() for char in content_to_check):
                valid_equations += 1

            # 3. Physical consistency (using PhysicsEngine)
            # Placeholder: In a real task, we'd parse the output for forces/reactions
            # For now, we simulate a check based on the presence of units
            if "N" in content_to_check and "m" in content_to_check:
                physically_consistent += 1

            # 4. Task accuracy (exact match or similar)
            if target and target.lower() in output.lower():
                task_correct += 1

        metrics = {
            'avg_token_count': total_tokens / total_count if total_count > 0 else 0,
            'equation_validity_rate': valid_equations / total_count if total_count > 0 else 0,
            'physical_consistency_rate': physically_consistent / total_count if total_count > 0 else 0,
            'task_accuracy': task_correct / total_count if total_count > 0 else 0
        }
        
        return metrics

    def generate_report(self) -> pd.DataFrame:
        """
        Generates a summary report comparing the results.
        """
        report_data = []
        for res in self.results:
            row = {'model_type': res['model_type']}
            row.update(res['metrics'])
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        if len(df) >= 2:
            # Calculate token reduction if we have both baseline and bdrl
            baseline_tokens = df[df['model_type'] == 'baseline']['avg_token_count'].values
            bdrl_tokens = df[df['model_type'] == 'bdrl']['avg_token_count'].values
            if len(baseline_tokens) > 0 and len(bdrl_tokens) > 0:
                reduction = (baseline_tokens[0] - bdrl_tokens[0]) / baseline_tokens[0]
                print(f"Token reduction: {reduction:.2%}")
                
        return df
