import random
import time
from typing import List, Dict, Any, Optional, Callable
from .agent import CADTraceAgent
from .payload import ReasoningPayload

class AdversarialBenchmark:
    """
    Evaluates the CAD-TRACE system against synthetic reasoning chains
    containing deliberate drift points and adversarial noise.
    """

    def __init__(
        self,
        agent_config: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.agent_config = agent_config or {
            "resilience_threshold": 0.6,
            "adversarial_threshold": 0.5,
            "auto_heal": True
        }
        self.agent = CADTraceAgent(**self.agent_config)
        self.results: List[Dict[str, Any]] = []
        random.seed(seed)

    def generate_drifting_chain(
        self,
        length: int = 10,
        drift_index: int = 5,
        drift_intensity: float = 0.8,
        noise_level: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Generates a sequence of payloads where reasoning begins to drift
        at a specific index.
        """
        chain = []
        root_intent = "Solve the complex causal inference problem X."
        
        # Base vector for intent
        intent_vector = [1.0] * 10
        
        for i in range(length):
            is_drifting = i >= drift_index
            
            # Simulate semantic vector drift
            if not is_drifting:
                # Normal reasoning: small noise
                vector = [v + random.uniform(-noise_level, noise_level) for v in intent_vector]
                content = f"Reasoning step {i}: Progressing towards goal."
            else:
                # Drift logic: move vector away from intent
                shift = drift_intensity * (i - drift_index + 1)
                vector = [v + random.uniform(shift * 0.5, shift) for v in intent_vector]
                content = f"Reasoning step {i}: [DRIFT] Diverging from original intent."

            payload = ReasoningPayload(
                source_id=f"agent-{i}",
                content=content,
                semantic_vector=vector
            )
            
            chain.append({
                "payload": payload,
                "is_drift_ground_truth": is_drifting,
                "is_origin": i == drift_index
            })
            
        return chain, root_intent

    def run_test_case(
        self,
        length: int = 10,
        drift_index: int = 5,
        drift_intensity: float = 0.8,
        noise_level: float = 0.2
    ) -> Dict[str, Any]:
        """
        Runs a single benchmark test case.
        """
        # Reset agent for new test case
        self.agent = CADTraceAgent(**self.agent_config)
        chain, root_intent = self.generate_drifting_chain(
            length, drift_index, drift_intensity, noise_level
        )
        
        parent_ids = []
        start_time = time.time()
        
        detections = []
        drift_origin_detected = None
        healing_events = 0
        
        # Mock regenerator for healing
        def mock_regenerator(intent: str, context: List[ReasoningPayload]) -> ReasoningPayload:
            return ReasoningPayload(
                source_id="healer-agent",
                content=f"Fixed reasoning based on intent: {intent}",
                semantic_vector=[1.0] * 10 # Back to intent
            )

        for item in chain:
            payload = item["payload"]
            res = self.agent.process_interaction(
                payload, 
                parent_ids=parent_ids,
                root_intent=root_intent,
                regenerator_fn=mock_regenerator
            )
            
            node_id = res["node_id"]
            parent_ids = [node_id]
            
            is_detected = res["system_status"] in ["drifting", "healed", "pruned"]
            detections.append(is_detected)
            
            if res["healing_performed"]:
                healing_events += 1
            
            # Check if system identified drift origin correctly
            summary = self.agent.get_system_summary()
            if summary["drift_origin"] and not drift_origin_detected:
                # Check if it matches our ground truth drift_index
                origin_node_payload = self.agent.tracker.get_payload(summary["drift_origin"])
                if origin_node_payload and f"step {drift_index}" in origin_node_payload.content:
                    drift_origin_detected = True

        duration = time.time() - start_time
        
        # Metrics Calculation
        tp = 0 # True Positives
        fp = 0 # False Positives
        fn = 0 # False Negatives
        tn = 0 # True Negatives
        
        for i, detected in enumerate(detections):
            gt = chain[i]["is_drift_ground_truth"]
            if detected and gt: tp += 1
            elif detected and not gt: fp += 1
            elif not detected and gt: fn += 1
            else: tn += 1

        # Token Savings approximation:
        # Full Restart (Standard) = 2 * length (One full failed run, one full corrected run)
        # CAD-TRACE = drift_index (before drift) + pruning overhead + (length - drift_index) (corrected part)
        # This is a bit simplistic but works for a benchmark comparison.
        total_steps_restart = 2 * length
        total_steps_cad = length + (1 if healing_events > 0 else 0) # Just processed once with detection
        token_savings = (total_steps_restart - total_steps_cad) / total_steps_restart if length > 0 else 0
        
        result = {
            "test_config": {
                "length": length,
                "drift_index": drift_index,
                "drift_intensity": drift_intensity
            },
            "metrics": {
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "drift_origin_found": drift_origin_detected or False,
                "healing_events": healing_events,
                "token_savings": token_savings,
                "execution_time": duration
            }
        }
        
        self.results.append(result)
        return result

    def batch_benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        """
        Runs multiple iterations with varying drift parameters.
        """
        for _ in range(iterations):
            length = random.randint(5, 15)
            drift_index = random.randint(2, length - 2)
            self.run_test_case(length=length, drift_index=drift_index)
            
        # Aggregate results
        avg_precision = sum(r["metrics"]["precision"] for r in self.results) / len(self.results)
        avg_recall = sum(r["metrics"]["recall"] for r in self.results) / len(self.results)
        avg_fpr = sum(r["metrics"]["false_positive_rate"] for r in self.results) / len(self.results)
        avg_savings = sum(r["metrics"]["token_savings"] for r in self.results) / len(self.results)
        origin_accuracy = sum(1 for r in self.results if r["metrics"]["drift_origin_found"]) / len(self.results)
        
        return {
            "iterations": iterations,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_false_positive_rate": avg_fpr,
            "avg_token_savings": avg_savings,
            "origin_pinpoint_accuracy": origin_accuracy
        }
