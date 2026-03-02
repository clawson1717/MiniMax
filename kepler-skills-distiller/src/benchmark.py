"""
Equation Discovery Benchmark

Evaluates the expert (KeplerAgent) vs. distilled apprentice on
symbolic regression benchmarks including Feynman equations.
"""

import json
import math
import random
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class EquationProblem:
    """A single equation discovery problem."""
    
    name: str
    category: str  # e.g., "physics", "mathematics", "chemistry"
    variables: List[str]
    equation: str  # Symbolic form, e.g., "F = m * a"
    data_points: List[Dict[str, float]]
    complexity: int  # 1-5 scale
    description: str = ""
    
    def evaluate(self, values: Dict[str, float]) -> float:
        """Evaluate the equation with given variable values."""
        # Create a safe evaluation context
        context = {k: v for k, v in values.items()}
        context.update({
            "sin": math.sin,
            "cos": math.cos,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
        })
        
        # Parse and evaluate
        # Replace common notation with Python
        expr = self.equation
        expr = expr.replace("^", "**")
        
        try:
            return eval(expr, {"__builtins__": {}}, context)
        except Exception:
            return float("nan")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    problem_name: str
    model_type: str  # "expert" or "apprentice"
    predicted_equation: str
    exact_match: bool
    relative_error: float
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""
    
    timestamp: str
    total_problems: int
    expert_results: List[BenchmarkResult]
    apprentice_results: List[BenchmarkResult]
    expert_accuracy: float
    apprentice_accuracy: float
    expert_avg_mae: float
    apprentice_avg_mae: float
    expert_avg_time: float
    apprentice_avg_time: float
    improvement_ratio: float  # apprentice/expert performance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_problems": self.total_problems,
                "expert_accuracy": self.expert_accuracy,
                "apprentice_accuracy": self.apprentice_accuracy,
                "expert_avg_mae": self.expert_avg_mae,
                "apprentice_avg_mae": self.apprentice_avg_mae,
                "expert_avg_time": self.expert_avg_time,
                "apprentice_avg_time": self.apprentice_avg_time,
                "improvement_ratio": self.improvement_ratio,
            },
            "expert_results": [
                {
                    "problem": r.problem_name,
                    "exact_match": r.exact_match,
                    "mae": r.mae,
                    "time": r.time_seconds,
                }
                for r in self.expert_results
            ],
            "apprentice_results": [
                {
                    "problem": r.problem_name,
                    "exact_match": r.exact_match,
                    "mae": r.mae,
                    "time": r.time_seconds,
                }
                for r in self.apprentice_results
            ],
        }


class FeynmanDataset:
    """
    Feynman Symbolic Regression Benchmark.
    
    A collection of physics equations from the Feynman Lectures on Physics,
    commonly used for symbolic regression benchmarking.
    """
    
    def __init__(self):
        self.equations = self._create_feynman_equations()
    
    def _create_feynman_equations(self) -> List[EquationProblem]:
        """Create the Feynman equation benchmark set."""
        equations = []
        
        # Classical Mechanics
        equations.append(EquationProblem(
            name="newton_second_law",
            category="physics",
            variables=["F", "m", "a"],
            equation="F - m * a",
            data_points=self._generate_data("F - m * a", ["F", "m", "a"]),
            complexity=1,
            description="Newton's Second Law: F = ma",
        ))
        
        equations.append(EquationProblem(
            name="kinetic_energy",
            category="physics",
            variables=["E", "m", "v"],
            equation="E - 0.5 * m * v**2",
            data_points=self._generate_data("E - 0.5 * m * v**2", ["E", "m", "v"]),
            complexity=1,
            description="Kinetic Energy: E = (1/2)mv²",
        ))
        
        equations.append(EquationProblem(
            name="gravitational_force",
            category="physics",
            variables=["F", "G", "m1", "m2", "r"],
            equation="F - G * m1 * m2 / r**2",
            data_points=self._generate_data("F - G * m1 * m2 / r**2", ["F", "G", "m1", "m2", "r"]),
            complexity=2,
            description="Newton's Law of Gravitation: F = Gm₁m₂/r²",
        ))
        
        equations.append(EquationProblem(
            name="wave_equation",
            category="physics",
            variables=["v", "f", "lambda"],
            equation="v - f * lambda",
            data_points=self._generate_data("v - f * lambda", ["v", "f", "lambda"]),
            complexity=1,
            description="Wave equation: v = fλ",
        ))
        
        equations.append(EquationProblem(
            name="momentum",
            category="physics",
            variables=["p", "m", "v"],
            equation="p - m * v",
            data_points=self._generate_data("p - m * v", ["p", "m", "v"]),
            complexity=1,
            description="Momentum: p = mv",
        ))
        
        # Electromagnetism
        equations.append(EquationProblem(
            name="coulomb_law",
            category="physics",
            variables=["F", "k", "q1", "q2", "r"],
            equation="F - k * q1 * q2 / r**2",
            data_points=self._generate_data("F - k * q1 * q2 / r**2", ["F", "k", "q1", "q2", "r"]),
            complexity=2,
            description="Coulomb's Law: F = kq₁q₂/r²",
        ))
        
        equations.append(EquationProblem(
            name="electric_power",
            category="physics",
            variables=["P", "V", "I"],
            equation="P - V * I",
            data_points=self._generate_data("P - V * I", ["P", "V", "I"]),
            complexity=1,
            description="Electric Power: P = VI",
        ))
        
        equations.append(EquationProblem(
            name="ohms_law",
            category="physics",
            variables=["V", "I", "R"],
            equation="V - I * R",
            data_points=self._generate_data("V - I * R", ["V", "I", "R"]),
            complexity=1,
            description="Ohm's Law: V = IR",
        ))
        
        # Thermodynamics
        equations.append(EquationProblem(
            name="ideal_gas",
            category="physics",
            variables=["P", "V", "n", "R", "T"],
            equation="P * V - n * R * T",
            data_points=self._generate_data("P * V - n * R * T", ["P", "V", "n", "R", "T"]),
            complexity=2,
            description="Ideal Gas Law: PV = nRT",
        ))
        
        equations.append(EquationProblem(
            name="thermal_energy",
            category="physics",
            variables=["Q", "m", "c", "delta_T"],
            equation="Q - m * c * delta_T",
            data_points=self._generate_data("Q - m * c * delta_T", ["Q", "m", "c", "delta_T"]),
            complexity=1,
            description="Thermal Energy: Q = mcΔT",
        ))
        
        # Mathematics
        equations.append(EquationProblem(
            name="circle_area",
            category="mathematics",
            variables=["A", "pi", "r"],
            equation="A - pi * r**2",
            data_points=self._generate_data("A - pi * r**2", ["A", "pi", "r"]),
            complexity=1,
            description="Circle Area: A = πr²",
        ))
        
        equations.append(EquationProblem(
            name="sphere_volume",
            category="mathematics",
            variables=["V", "pi", "r"],
            equation="V - (4/3) * pi * r**3",
            data_points=self._generate_data("V - (4/3) * pi * r**3", ["V", "pi", "r"]),
            complexity=2,
            description="Sphere Volume: V = (4/3)πr³",
        ))
        
        equations.append(EquationProblem(
            name="pythagorean",
            category="mathematics",
            variables=["c", "a", "b"],
            equation="c**2 - a**2 - b**2",
            data_points=self._generate_data("c**2 - a**2 - b**2", ["c", "a", "b"]),
            complexity=1,
            description="Pythagorean Theorem: c² = a² + b²",
        ))
        
        return equations
    
    def _generate_data(
        self,
        equation: str,
        variables: List[str],
        n_points: int = 100,
    ) -> List[Dict[str, float]]:
        """Generate random data points that satisfy the equation."""
        data = []
        
        for _ in range(n_points):
            point = {}
            
            # Generate random values for independent variables
            for var in variables:
                if var in ["pi", "e"]:
                    continue  # Constants
                point[var] = random.uniform(0.1, 10.0)
            
            # Set constants
            if "pi" in variables:
                point["pi"] = math.pi
            if "e" in variables:
                point["e"] = math.e
            
            data.append(point)
        
        return data
    
    def get_problems(
        self,
        categories: Optional[List[str]] = None,
        max_complexity: Optional[int] = None,
    ) -> List[EquationProblem]:
        """
        Get filtered list of problems.
        
        Args:
            categories: Filter by categories (e.g., ["physics"])
            max_complexity: Maximum complexity level (1-5)
            
        Returns:
            List of matching EquationProblem instances.
        """
        problems = self.equations
        
        if categories:
            problems = [p for p in problems if p.category in categories]
        
        if max_complexity is not None:
            problems = [p for p in problems if p.complexity <= max_complexity]
        
        return problems


class EquationDiscoveryBenchmark:
    """
    Benchmark runner for equation discovery.
    
    Evaluates expert and apprentice models on symbolic regression tasks.
    """
    
    def __init__(
        self,
        dataset: Optional[FeynmanDataset] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the benchmark.
        
        Args:
            dataset: Dataset to use. Defaults to FeynmanDataset.
            output_dir: Directory to save results.
        """
        self.dataset = dataset or FeynmanDataset()
        self.output_dir = Path(output_dir) if output_dir else None
    
    def evaluate_model(
        self,
        model: Callable,
        model_type: str,
        problems: List[EquationProblem],
    ) -> List[BenchmarkResult]:
        """
        Evaluate a model on a set of problems.
        
        Args:
            model: Model callable that takes data and returns an equation.
            model_type: "expert" or "apprentice"
            problems: List of problems to evaluate.
            
        Returns:
            List of BenchmarkResult instances.
        """
        results = []
        
        for problem in problems:
            start_time = time.time()
            
            try:
                # Model should take data points and return predicted equation
                predicted = model(problem.data_points, problem.variables)
                
                # Parse predicted equation
                if isinstance(predicted, dict):
                    predicted_eq = predicted.get("equation", "")
                elif isinstance(predicted, str):
                    predicted_eq = predicted
                else:
                    predicted_eq = ""
                
                # Calculate metrics
                exact_match = self._check_exact_match(predicted_eq, problem.equation)
                mae, rmse = self._calculate_errors(predicted_eq, problem)
                relative_error = mae / (self._get_mean_magnitude(problem) + 1e-10)
                
            except Exception as e:
                predicted_eq = f"ERROR: {str(e)}"
                exact_match = False
                mae = float("inf")
                rmse = float("inf")
                relative_error = float("inf")
            
            elapsed = time.time() - start_time
            
            result = BenchmarkResult(
                problem_name=problem.name,
                model_type=model_type,
                predicted_equation=predicted_eq,
                exact_match=exact_match,
                relative_error=relative_error,
                mae=mae,
                rmse=rmse,
                time_seconds=elapsed,
            )
            
            results.append(result)
        
        return results
    
    def _check_exact_match(self, predicted: str, target: str) -> bool:
        """Check if predicted equation matches target (ignoring whitespace)."""
        pred_clean = predicted.replace(" ", "").lower()
        target_clean = target.replace(" ", "").lower()
        return pred_clean == target_clean
    
    def _calculate_errors(
        self,
        predicted_eq: str,
        problem: EquationProblem,
    ) -> Tuple[float, float]:
        """Calculate MAE and RMSE for predicted equation."""
        errors = []
        
        for point in problem.data_points[:20]:  # Sample points
            try:
                # Evaluate target
                target_value = problem.evaluate(point)
                
                # Evaluate predicted
                context = {k: v for k, v in point.items()}
                context.update({
                    "sin": math.sin, "cos": math.cos,
                    "exp": math.exp, "log": math.log,
                    "sqrt": math.sqrt, "abs": abs,
                    "pi": math.pi, "e": math.e,
                })
                expr = predicted_eq.replace("^", "**")
                pred_value = eval(expr, {"__builtins__": {}}, context)
                
                if not math.isnan(target_value) and not math.isnan(pred_value):
                    errors.append(abs(pred_value - target_value))
                    
            except Exception:
                continue
        
        if not errors:
            return float("inf"), float("inf")
        
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        
        return mae, rmse
    
    def _get_mean_magnitude(self, problem: EquationProblem) -> float:
        """Get mean magnitude of target values."""
        values = []
        for point in problem.data_points[:20]:
            val = problem.evaluate(point)
            if not math.isnan(val):
                values.append(abs(val))
        
        return sum(values) / len(values) if values else 1.0
    
    def run_benchmark(
        self,
        expert_model: Callable,
        apprentice_model: Optional[Callable] = None,
        categories: Optional[List[str]] = None,
        max_complexity: Optional[int] = None,
    ) -> BenchmarkReport:
        """
        Run full benchmark comparing expert and apprentice.
        
        Args:
            expert_model: Expert model (KeplerAgent).
            apprentice_model: Optional apprentice model.
            categories: Filter by categories.
            max_complexity: Maximum complexity.
            
        Returns:
            BenchmarkReport with results.
        """
        problems = self.dataset.get_problems(categories, max_complexity)
        
        # Evaluate expert
        expert_results = self.evaluate_model(expert_model, "expert", problems)
        
        # Evaluate apprentice if provided
        apprentice_results = []
        if apprentice_model:
            apprentice_results = self.evaluate_model(
                apprentice_model, "apprentice", problems
            )
        
        # Calculate summary statistics
        expert_accuracy = sum(1 for r in expert_results if r.exact_match) / len(expert_results)
        apprentice_accuracy = sum(1 for r in apprentice_results if r.exact_match) / len(apprentice_results) if apprentice_results else 0.0
        
        expert_avg_mae = sum(r.mae for r in expert_results if r.mae != float("inf")) / max(1, sum(1 for r in expert_results if r.mae != float("inf")))
        apprentice_avg_mae = sum(r.mae for r in apprentice_results if r.mae != float("inf")) / max(1, sum(1 for r in apprentice_results if r.mae != float("inf"))) if apprentice_results else 0.0
        
        expert_avg_time = sum(r.time_seconds for r in expert_results) / len(expert_results)
        apprentice_avg_time = sum(r.time_seconds for r in apprentice_results) / len(apprentice_results) if apprentice_results else 0.0
        
        # Calculate improvement ratio (apprentice should approach expert)
        if apprentice_results and expert_accuracy > 0:
            improvement_ratio = apprentice_accuracy / expert_accuracy
        else:
            improvement_ratio = 0.0
        
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_problems=len(problems),
            expert_results=expert_results,
            apprentice_results=apprentice_results,
            expert_accuracy=expert_accuracy,
            apprentice_accuracy=apprentice_accuracy,
            expert_avg_mae=expert_avg_mae,
            apprentice_avg_mae=apprentice_avg_mae,
            expert_avg_time=expert_avg_time,
            apprentice_avg_time=apprentice_avg_time,
            improvement_ratio=improvement_ratio,
        )
        
        # Save if output directory specified
        if self.output_dir:
            self._save_report(report)
        
        return report
    
    def _save_report(self, report: BenchmarkReport) -> None:
        """Save benchmark report to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"benchmark_{report.timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


def create_mock_expert() -> Callable:
    """Create a mock expert model for testing."""
    def expert_model(data_points, variables):
        # Mock: return a simple linear combination
        return {"equation": " + ".join(variables[:2]) + " - 1.0"}
    return expert_model


def create_mock_apprentice() -> Callable:
    """Create a mock apprentice model for testing."""
    def apprentice_model(data_points, variables):
        # Mock: return less accurate equation
        return {"equation": variables[0] + " * 2"}
    return apprentice_model


def run_quick_benchmark() -> BenchmarkReport:
    """Run a quick benchmark with mock models."""
    benchmark = EquationDiscoveryBenchmark()
    expert = create_mock_expert()
    apprentice = create_mock_apprentice()
    
    return benchmark.run_benchmark(
        expert_model=expert,
        apprentice_model=apprentice,
        max_complexity=2,
    )


if __name__ == "__main__":
    report = run_quick_benchmark()
    print(f"\nBenchmark Results:")
    print(f"  Expert Accuracy: {report.expert_accuracy:.1%}")
    print(f"  Apprentice Accuracy: {report.apprentice_accuracy:.1%}")
    print(f"  Expert Avg MAE: {report.expert_avg_mae:.4f}")
    print(f"  Apprentice Avg MAE: {report.apprentice_avg_mae:.4f}")
    print(f"  Improvement Ratio: {report.improvement_ratio:.2%}")