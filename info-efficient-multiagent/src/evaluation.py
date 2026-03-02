"""
Comparative Evaluation of Routing Strategies

Compares three routing strategies:
1. Info-efficient routing: Route based on agent information capacity
2. Uniform routing: Equal compute/attention to all agents
3. Uncertainty-based routing: Route based on agent uncertainty (inverse confidence)

Provides statistical analysis and comparison reports.
"""

import random
import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math

from .benchmarks import (
    BenchmarkTask,
    BenchmarkResults,
    EvaluationResult,
    TaskDataset,
    BenchmarkType,
    Difficulty,
    MultiAgentBenchmarkRunner,
)
from .routing import RouteMode, MessageRouter
from .environment import MultiAgentEnvironment, AgentRole, MessageType
from .agent import ReasoningAgent, AgentConfig
from .capacity import InformationCapacityEstimator


class RoutingStrategy(Enum):
    """Routing strategies for comparative evaluation."""
    INFO_EFFICIENT = "info_efficient"  # Capacity-weighted routing
    UNIFORM = "uniform"  # Equal compute/attention to all agents
    UNCERTAINTY_BASED = "uncertainty_based"  # Route based on uncertainty (inverse confidence)


@dataclass
class RoutingMetrics:
    """Metrics collected during routing evaluation."""
    total_messages: int = 0
    total_tokens_used: int = 0
    total_capacity_used: float = 0.0
    avg_response_time_ms: float = 0.0
    routing_overhead_ms: float = 0.0
    messages_by_agent: Dict[str, int] = field(default_factory=dict)
    capacity_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Results for a single routing strategy evaluation."""
    strategy: RoutingStrategy
    benchmark_results: BenchmarkResults
    routing_metrics: RoutingMetrics
    per_task_metrics: List[Dict[str, Any]] = field(default_factory=list)
    agent_utilization: Dict[str, float] = field(default_factory=dict)
    compute_efficiency: float = 0.0
    response_quality_score: float = 0.0


@dataclass
class ComparisonReport:
    """Complete comparison report across all strategies."""
    strategies_compared: List[RoutingStrategy]
    results_by_strategy: Dict[str, StrategyResult]
    accuracy_comparison: Dict[str, float]
    token_efficiency_comparison: Dict[str, float]
    quality_comparison: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, Any]]
    winner_by_metric: Dict[str, str]
    summary: str
    recommendations: List[str]


class MockModel:
    """Mock model for generating responses during evaluation."""
    
    def __init__(self, quality: float = 0.8, seed: Optional[int] = None):
        self.quality = quality
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate a mock response based on prompt type."""
        prompt_lower = prompt.lower()
        
        # Check for math operators first (before "what is" QA pattern)
        if "+" in prompt_lower or "-" in prompt_lower or "×" in prompt_lower or "÷" in prompt_lower or "*" in prompt_lower or "/" in prompt_lower:
            return self._handle_math(prompt)
        elif "what is" in prompt_lower or "capital" in prompt_lower:
            return self._handle_qa(prompt)
        elif "is this" in prompt_lower or "valid" in prompt_lower or "to what" in prompt_lower:
            return self._handle_reasoning(prompt)
        else:
            return self._handle_general(prompt)
    
    def _handle_qa(self, prompt: str) -> str:
        """Handle QA tasks."""
        qa_answers = {
            "capital of france": "Paris",
            "capital of australia": "Canberra",
            "wrote 'romeo and juliet'": "Shakespeare",
            "largest planet": "Jupiter",
            "chemical symbol for gold": "Au",
            "tallest mountain": "Mount Everest",
            "painted the mona lisa": "Leonardo da Vinci",
            "speed of light": "299792458 m/s",
            "world war ii ended": "1945",
            "general relativity": "Einstein",
        }
        for key, answer in qa_answers.items():
            if key in prompt.lower():
                if random.random() < self.quality:
                    return f"The answer is {answer}."
                else:
                    return f"I believe the answer might be related to {answer.lower()}."
        return "I need more context to answer this question accurately."
    
    def _handle_math(self, prompt: str) -> str:
        """Handle math tasks."""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', prompt)
        
        if "+" in prompt and len(numbers) >= 2:
            result = float(numbers[0]) + float(numbers[1])
        elif "-" in prompt and len(numbers) >= 2:
            result = float(numbers[0]) - float(numbers[1])
        elif "×" in prompt or "*" in prompt:
            if len(numbers) >= 2:
                result = float(numbers[0]) * float(numbers[1])
            else:
                result = 0
        elif "÷" in prompt or "/" in prompt:
            if len(numbers) >= 2 and float(numbers[1]) != 0:
                result = float(numbers[0]) / float(numbers[1])
            else:
                result = 0
        elif "square root" in prompt.lower() and numbers:
            result = math.sqrt(float(numbers[0]))
        elif "²" in prompt and numbers:
            result = float(numbers[0]) ** 2
        elif "%" in prompt and len(numbers) >= 2:
            result = float(numbers[0]) * float(numbers[1]) / 100
        else:
            result = 0
        
        # Add noise based on quality
        if random.random() > self.quality:
            result += random.uniform(-5, 5)
        
        return f"The answer is {result:.0f}."
    
    def _handle_reasoning(self, prompt: str) -> str:
        """Handle reasoning tasks."""
        reasoning_answers = {
            "all cats are mammals": "yes",
            "penguins can fly": "no",
            "ground is wet": "no",
            "did not pass": "yes",
            "hot is to cold": "dark",
            "doctor is to hospital": "school",
            "book is to reading": "eating",
            "planet is to solar system": "atom",
            "alice is taller": "yes",
            "all a are b": "no",
        }
        
        for key, answer in reasoning_answers.items():
            if key in prompt.lower():
                if random.random() < self.quality:
                    return f"The answer is {answer}. This follows from logical analysis."
                else:
                    opposite = "no" if answer == "yes" else "yes"
                    return f"I think the answer might be {opposite}."
        
        return "Based on careful reasoning, I would need more information to conclude."
    
    def _handle_general(self, prompt: str) -> str:
        """Handle general prompts."""
        responses = [
            "This requires careful analysis of the given information.",
            "Based on my understanding, there are multiple factors to consider.",
            "The question touches on several important aspects worth examining.",
        ]
        return random.choice(responses)


class UniformRouter:
    """
    Router that distributes messages uniformly across all agents.
    
    This represents a baseline where all agents receive equal compute
    and attention, regardless of their capacity or uncertainty.
    """
    
    def __init__(self, environment: MultiAgentEnvironment):
        self.environment = environment
        self.routing_history: List[Dict[str, Any]] = []
        self.stats = {
            "total_messages": 0,
            "messages_by_agent": defaultdict(int),
        }
    
    def route(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.TASK,
        exclude_sender: bool = True,
    ) -> List[Any]:
        """Route message uniformly to all agents."""
        target_agents = [
            aid for aid in self.environment.agents.keys()
            if not exclude_sender or aid != sender_id
        ]
        
        if not target_agents:
            return []
        
        messages = []
        for target_id in target_agents:
            msg = self.environment.send_message(
                sender_id=sender_id,
                receiver_id=target_id,
                message_type=message_type,
                content=content,
                metadata={"routing_mode": "uniform"},
            )
            messages.append(msg)
            self.stats["messages_by_agent"][target_id] += 1
        
        self.stats["total_messages"] += len(messages)
        self.routing_history.append({
            "sender": sender_id,
            "targets": target_agents,
            "mode": "uniform",
        })
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)


class UncertaintyBasedRouter:
    """
    Router that distributes messages based on agent uncertainty.
    
    Agents with higher uncertainty (lower confidence) receive more
    attention/compute, as they need more help. This is the inverse
    of capacity-weighted routing.
    """
    
    def __init__(
        self,
        environment: MultiAgentEnvironment,
        temperature: float = 1.0,
    ):
        self.environment = environment
        self.temperature = temperature
        self.routing_history: List[Dict[str, Any]] = []
        self.stats = {
            "total_messages": 0,
            "messages_by_agent": defaultdict(int),
        }
        self._agent_uncertainties: Dict[str, float] = {}
    
    def update_uncertainty(self, agent_id: str, confidence: float) -> None:
        """Update an agent's uncertainty (1 - confidence)."""
        self._agent_uncertainties[agent_id] = 1.0 - confidence
    
    def get_uncertainty(self, agent_id: str) -> float:
        """Get an agent's uncertainty level."""
        return self._agent_uncertainties.get(agent_id, 0.5)
    
    def route(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.TASK,
        exclude_sender: bool = True,
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """Route message based on uncertainty weights."""
        eligible_agents = [
            aid for aid in self.environment.agents.keys()
            if not exclude_sender or aid != sender_id
        ]
        
        if not eligible_agents:
            return []
        
        # Compute uncertainty weights (higher uncertainty = higher weight)
        weights = {}
        for aid in eligible_agents:
            uncertainty = self.get_uncertainty(aid)
            # Use capacity as fallback if no uncertainty info
            if uncertainty == 0.5 and aid in self.environment.agent_states:
                # Inverse of capacity as uncertainty proxy
                cap = self.environment.agent_states[aid].capacity
                uncertainty = 1.0 - min(cap, 1.0)
            weights[aid] = uncertainty
        
        # Apply temperature scaling (softmax)
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Apply top_k if specified
        if top_k is not None:
            sorted_agents = sorted(
                eligible_agents,
                key=lambda x: weights.get(x, 0),
                reverse=True
            )
            eligible_agents = sorted_agents[:top_k]
        
        # Route to all eligible agents with uncertainty-based priority
        messages = []
        for agent_id in eligible_agents:
            msg = self.environment.send_message(
                sender_id=sender_id,
                receiver_id=agent_id,
                message_type=message_type,
                content=content,
                metadata={
                    "routing_mode": "uncertainty_based",
                    "uncertainty_weight": weights.get(agent_id, 0),
                },
            )
            messages.append(msg)
            self.stats["messages_by_agent"][agent_id] += 1
        
        self.stats["total_messages"] += len(messages)
        self.routing_history.append({
            "sender": sender_id,
            "targets": eligible_agents,
            "mode": "uncertainty_based",
            "weights": weights,
        })
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **dict(self.stats),
            "agent_uncertainties": dict(self._agent_uncertainties),
        }


class ComparativeEvaluator:
    """
    Evaluates and compares different routing strategies.
    
    Runs benchmarks with each routing strategy and collects metrics
    for statistical comparison.
    """
    
    def __init__(
        self,
        environment: MultiAgentEnvironment,
        agents: List[ReasoningAgent],
        seed: Optional[int] = None,
    ):
        """
        Initialize the comparative evaluator.
        
        Args:
            environment: Multi-agent environment
            agents: List of reasoning agents
            seed: Random seed for reproducibility
        """
        self.environment = environment
        self.agents = agents
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create routing instances
        self.capacity_router = MessageRouter(
            environment,
            default_mode=RouteMode.CAPACITY_WEIGHTED,
        )
        self.uniform_router = UniformRouter(environment)
        self.uncertainty_router = UncertaintyBasedRouter(environment)
        
        # Benchmark runner
        self.runner = MultiAgentBenchmarkRunner()
        
        # Results storage
        self.results: Dict[str, StrategyResult] = {}
    
    def setup_agents(self, capacities: Optional[List[float]] = None) -> None:
        """Setup agent capacities in the environment."""
        for i, agent in enumerate(self.agents):
            capacity = capacities[i] if capacities and i < len(capacities) else random.uniform(0.3, 1.0)
            if agent.agent_id in self.environment.agent_states:
                self.environment.agent_states[agent.agent_id].capacity = capacity
        
        # Initialize uncertainty router with inverse capacities
        for agent_id, state in self.environment.agent_states.items():
            uncertainty = 1.0 - min(state.capacity, 1.0)
            self.uncertainty_router.update_uncertainty(agent_id, 1.0 - state.capacity)
    
    def evaluate_strategy(
        self,
        strategy: RoutingStrategy,
        tasks: TaskDataset,
        model_quality: float = 0.8,
    ) -> StrategyResult:
        """
        Evaluate a single routing strategy on a set of tasks.
        
        Args:
            strategy: The routing strategy to evaluate
            tasks: Dataset of benchmark tasks
            model_quality: Quality of the mock model (0-1)
            
        Returns:
            StrategyResult with all metrics
        """
        # Reset environment for fair comparison
        self.environment.reset()
        
        # Create mock model
        model = MockModel(quality=model_quality, seed=self.seed)
        
        # Track metrics
        routing_metrics = RoutingMetrics()
        per_task_metrics = []
        all_results = []
        
        total_tokens = 0
        total_capacity = 0.0
        total_time = 0.0
        
        for task in tasks:
            start_time = time.time()
            
            # Route based on strategy
            routing_start = time.time()
            if strategy == RoutingStrategy.INFO_EFFICIENT:
                messages = self.capacity_router.route(
                    sender_id="coordinator",
                    content=task.prompt,
                    message_type=MessageType.TASK,
                    top_k=3,
                )
            elif strategy == RoutingStrategy.UNIFORM:
                messages = self.uniform_router.route(
                    sender_id="coordinator",
                    content=task.prompt,
                    message_type=MessageType.TASK,
                )
            else:  # UNCERTAINTY_BASED
                messages = self.uncertainty_router.route(
                    sender_id="coordinator",
                    content=task.prompt,
                    message_type=MessageType.TASK,
                    top_k=3,
                )
            
            routing_time = (time.time() - routing_start) * 1000
            
            # Generate response using mock model
            response = model(task.prompt)
            
            # Simulate token usage (based on response length)
            tokens = len(response.split()) + random.randint(10, 30)
            total_tokens += tokens
            
            # Simulate capacity usage
            if strategy == RoutingStrategy.INFO_EFFICIENT:
                cap_used = sum(
                    self.environment.agent_states.get(aid, type('', (), {'capacity': 0.5})()).capacity
                    for aid in [m.receiver_id for m in messages if m.receiver_id]
                )
            elif strategy == RoutingStrategy.UNIFORM:
                cap_used = len(messages) * 0.5  # Average capacity
            else:
                cap_used = sum(
                    self.uncertainty_router.get_uncertainty(aid)
                    for aid in [m.receiver_id for m in messages if m.receiver_id]
                )
            total_capacity += cap_used
            
            # Evaluate response
            eval_result = self.runner.evaluate_response(task, response, "coordinator")
            all_results.append(eval_result)
            
            elapsed_time = (time.time() - start_time) * 1000
            total_time += elapsed_time
            
            # Track per-task metrics
            per_task_metrics.append({
                "task_id": task.task_id,
                "is_correct": eval_result.is_correct,
                "score": eval_result.score,
                "tokens": tokens,
                "capacity_used": cap_used,
                "response_time_ms": elapsed_time,
                "routing_time_ms": routing_time,
            })
            
            # Update uncertainty for uncertainty-based routing
            if strategy == RoutingStrategy.UNCERTAINTY_BASED:
                confidence = eval_result.score if eval_result.score > 0 else 0.5
                self.uncertainty_router.update_uncertainty("coordinator", confidence)
        
        # Aggregate results
        correct = sum(1 for r in all_results if r.is_correct)
        total_score = sum(r.score for r in all_results)
        
        benchmark_results = BenchmarkResults(
            benchmark_name=f"{strategy.value}_benchmark",
            total_tasks=len(tasks),
            correct=correct,
            total_score=total_score,
            accuracy=correct / len(tasks) if tasks else 0,
            avg_response_time_ms=total_time / len(tasks) if tasks else 0,
            total_tokens_used=total_tokens,
            total_capacity_used=total_capacity,
            per_task_results=all_results,
        )
        
        # Compute routing metrics
        routing_metrics.total_messages = len(tasks) * 3  # Approximate
        routing_metrics.total_tokens_used = total_tokens
        routing_metrics.total_capacity_used = total_capacity
        routing_metrics.avg_response_time_ms = total_time / len(tasks) if tasks else 0
        
        # Get agent utilization
        if strategy == RoutingStrategy.INFO_EFFICIENT:
            stats = self.capacity_router.get_routing_stats()
            routing_metrics.messages_by_agent = stats.get("messages_by_receiver", {})
        elif strategy == RoutingStrategy.UNIFORM:
            stats = self.uniform_router.get_stats()
            routing_metrics.messages_by_agent = stats.get("messages_by_agent", {})
        else:
            stats = self.uncertainty_router.get_stats()
            routing_metrics.messages_by_agent = stats.get("messages_by_agent", {})
        
        # Compute agent utilization
        agent_utilization = {}
        for agent_id in self.environment.agents:
            agent_utilization[agent_id] = routing_metrics.messages_by_agent.get(agent_id, 0)
        
        # Normalize utilization
        total_msgs = sum(agent_utilization.values())
        if total_msgs > 0:
            agent_utilization = {k: v / total_msgs for k, v in agent_utilization.items()}
        
        # Compute efficiency metrics
        compute_efficiency = benchmark_results.accuracy / (total_tokens / 1000) if total_tokens > 0 else 0
        capacity_factor = total_capacity / (len(tasks) * 10) if len(tasks) > 0 else 0
        quality_score = benchmark_results.accuracy * 0.7 + (1 - capacity_factor) * 0.3
        
        return StrategyResult(
            strategy=strategy,
            benchmark_results=benchmark_results,
            routing_metrics=routing_metrics,
            per_task_metrics=per_task_metrics,
            agent_utilization=agent_utilization,
            compute_efficiency=compute_efficiency,
            response_quality_score=quality_score,
        )
    
    def run_comparison(
        self,
        tasks: Optional[TaskDataset] = None,
        model_quality: float = 0.8,
        agent_capacities: Optional[List[float]] = None,
    ) -> ComparisonReport:
        """
        Run full comparison of all routing strategies.
        
        Args:
            tasks: Dataset of tasks (uses default if None)
            model_quality: Quality of mock model (0-1)
            agent_capacities: Optional capacities for agents
            
        Returns:
            ComparisonReport with all results
        """
        # Get tasks
        if tasks is None:
            tasks = self.runner.get_all_tasks()
        
        # Setup agents
        self.setup_agents(agent_capacities)
        
        # Evaluate each strategy
        results = {}
        for strategy in RoutingStrategy:
            results[strategy.value] = self.evaluate_strategy(
                strategy=strategy,
                tasks=tasks,
                model_quality=model_quality,
            )
        
        self.results = results
        
        # Generate comparison report
        return self._generate_report(results)
    
    def _generate_report(
        self,
        results: Dict[str, StrategyResult],
    ) -> ComparisonReport:
        """Generate a comprehensive comparison report."""
        # Accuracy comparison
        accuracy_comparison = {
            name: result.benchmark_results.accuracy
            for name, result in results.items()
        }
        
        # Token efficiency comparison
        token_efficiency = {
            name: result.compute_efficiency
            for name, result in results.items()
        }
        
        # Quality comparison
        quality_comparison = {
            name: result.response_quality_score
            for name, result in results.items()
        }
        
        # Statistical tests
        statistical_tests = self._compute_statistical_tests(results)
        
        # Determine winners
        winner_by_metric = {
            "accuracy": max(accuracy_comparison.keys(), key=lambda k: accuracy_comparison[k]),
            "token_efficiency": max(token_efficiency.keys(), key=lambda k: token_efficiency[k]),
            "quality": max(quality_comparison.keys(), key=lambda k: quality_comparison[k]),
        }
        
        # Generate summary
        summary = self._generate_summary(results, winner_by_metric)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, winner_by_metric)
        
        return ComparisonReport(
            strategies_compared=list(RoutingStrategy),
            results_by_strategy=results,
            accuracy_comparison=accuracy_comparison,
            token_efficiency_comparison=token_efficiency,
            quality_comparison=quality_comparison,
            statistical_tests=statistical_tests,
            winner_by_metric=winner_by_metric,
            summary=summary,
            recommendations=recommendations,
        )
    
    def _compute_statistical_tests(
        self,
        results: Dict[str, StrategyResult],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistical tests comparing strategies."""
        tests = {}
        
        strategies = list(results.keys())
        
        # Paired t-test approximation (using per-task scores)
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                scores1 = [m["score"] for m in results[s1].per_task_metrics]
                scores2 = [m["score"] for m in results[s2].per_task_metrics]
                
                # Compute mean and std
                mean1, std1 = np.mean(scores1), np.std(scores1)
                mean2, std2 = np.mean(scores2), np.std(scores2)
                
                # Paired differences
                if len(scores1) == len(scores2):
                    diffs = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs)
                    n = len(diffs)
                    
                    # t-statistic
                    if std_diff > 0:
                        t_stat = mean_diff / (std_diff / np.sqrt(n))
                    else:
                        t_stat = 0
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    if pooled_std > 0:
                        cohens_d = (mean1 - mean2) / pooled_std
                    else:
                        cohens_d = 0
                else:
                    t_stat = 0
                    cohens_d = 0
                    mean_diff = mean1 - mean2
                
                test_key = f"{s1}_vs_{s2}"
                tests[test_key] = {
                    "mean_difference": mean_diff,
                    "t_statistic": t_stat,
                    "cohens_d": cohens_d,
                    "significant": abs(cohens_d) > 0.5,  # Medium effect size threshold
                    "winner": s1 if mean1 > mean2 else s2 if mean2 > mean1 else "tie",
                }
        
        return tests
    
    def _generate_summary(
        self,
        results: Dict[str, StrategyResult],
        winner_by_metric: Dict[str, str],
    ) -> str:
        """Generate a text summary of the comparison."""
        lines = ["Comparative Evaluation Summary", "=" * 40, ""]
        
        # Overall results
        for name, result in results.items():
            acc = result.benchmark_results.accuracy
            tokens = result.benchmark_results.total_tokens_used
            quality = result.response_quality_score
            lines.append(f"{name.upper()}:")
            lines.append(f"  Accuracy: {acc:.2%}")
            lines.append(f"  Tokens: {tokens}")
            lines.append(f"  Quality Score: {quality:.3f}")
            lines.append("")
        
        # Winners
        lines.append("Winners by Metric:")
        for metric, winner in winner_by_metric.items():
            lines.append(f"  {metric}: {winner}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(
        self,
        results: Dict[str, StrategyResult],
        winner_by_metric: Dict[str, str],
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        info_efficient = results.get(RoutingStrategy.INFO_EFFICIENT.value)
        uniform = results.get(RoutingStrategy.UNIFORM.value)
        uncertainty = results.get(RoutingStrategy.UNCERTAINTY_BASED.value)
        
        if info_efficient and uniform:
            if info_efficient.benchmark_results.accuracy > uniform.benchmark_results.accuracy:
                recommendations.append(
                    "Info-efficient routing outperforms uniform routing for accuracy. "
                    "Use capacity-weighted routing when agent capacity varies significantly."
                )
            elif uniform.benchmark_results.accuracy > info_efficient.benchmark_results.accuracy:
                recommendations.append(
                    "Uniform routing performs comparably or better. "
                    "Consider simpler routing when agent capacities are similar."
                )
        
        if info_efficient and uncertainty:
            if info_efficient.benchmark_results.accuracy > uncertainty.benchmark_results.accuracy:
                recommendations.append(
                    "Info-efficient routing outperforms uncertainty-based routing. "
                    "Capacity is a better signal than uncertainty for routing decisions."
                )
            else:
                recommendations.append(
                    "Uncertainty-based routing shows promise. "
                    "Consider combining capacity and uncertainty for hybrid routing."
                )
        
        # General recommendations
        recommendations.append(
            "Consider the tradeoff between accuracy and compute efficiency "
            "when selecting a routing strategy for production use."
        )
        
        return recommendations


def create_evaluation_environment(
    num_agents: int = 5,
    seed: Optional[int] = None,
) -> Tuple[MultiAgentEnvironment, List[ReasoningAgent]]:
    """
    Create an environment and agents for evaluation.
    
    Args:
        num_agents: Number of agents to create
        seed: Random seed
        
    Returns:
        Tuple of (environment, agents)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create environment
    env = MultiAgentEnvironment(
        name="evaluation-env",
        max_agents=num_agents + 2,
    )
    
    # Create agents with diverse specializations
    specializations_list = [
        ["math", "reasoning"],
        ["science", "qa"],
        ["reasoning", "language"],
        ["math", "coding"],
        ["general"],
    ][:num_agents]
    
    agents = []
    for i in range(num_agents):
        config = AgentConfig(
            agent_id=f"agent_{i}",
            specializations=specializations_list[i] if i < len(specializations_list) else ["general"],
        )
        agent = ReasoningAgent(config=config)
        agents.append(agent)
        
        # Register in environment
        role = AgentRole.WORKER if i < num_agents - 1 else AgentRole.EVALUATOR
        env.register_agent(
            agent_id=agent.agent_id,
            agent=agent,
            role=role,
            capacity=random.uniform(0.3, 1.0),
        )
    
    return env, agents


def run_quick_evaluation(
    num_agents: int = 3,
    num_tasks: int = 10,
    model_quality: float = 0.8,
    seed: int = 42,
) -> ComparisonReport:
    """
    Run a quick evaluation for testing purposes.
    
    Args:
        num_agents: Number of agents
        num_tasks: Number of tasks to evaluate
        model_quality: Quality of mock model
        seed: Random seed
        
    Returns:
        ComparisonReport
    """
    # Create environment and agents
    env, agents = create_evaluation_environment(num_agents, seed)
    
    # Create evaluator
    evaluator = ComparativeEvaluator(env, agents, seed)
    
    # Get tasks
    runner = MultiAgentBenchmarkRunner()
    all_tasks = runner.get_all_tasks()
    
    # Sample subset
    tasks = all_tasks.sample(num_tasks, seed=seed)
    
    # Run comparison
    return evaluator.run_comparison(
        tasks=tasks,
        model_quality=model_quality,
    )


if __name__ == "__main__":
    # Demo usage
    print("Running comparative evaluation...")
    
    report = run_quick_evaluation(
        num_agents=3,
        num_tasks=15,
        model_quality=0.8,
        seed=42,
    )
    
    print("\n" + report.summary)
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    print("\nStatistical Tests:")
    for test_name, test_results in report.statistical_tests.items():
        print(f"  {test_name}:")
        print(f"    Mean diff: {test_results['mean_difference']:.4f}")
        print(f"    Cohen's d: {test_results['cohens_d']:.4f}")
        print(f"    Significant: {test_results['significant']}")
        print(f"    Winner: {test_results['winner']}")
