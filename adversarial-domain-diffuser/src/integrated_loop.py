import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.adversarial_gen import AdversarialGenerator
from src.environment import AgentEnvironment
from src.diffusion import DiffusionPolicy
from src.omad import OMADOrchestrator
from src.grouping import EmbodimentGrouper
from src.reasoning_agent import DomainReasoningAgent

class IntegratedAdversarialLoop:
    """
    Orchestrates the adversarial loop between the generator and the multi-agent team.
    Iteratively closes the comprehension gap.
    """

    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        expert_reference: str,
        max_iterations: int = 3,
        model_client: Any = None
    ):
        """
        Initialize the integrated loop.

        Args:
            agent_configs: Configuration for the reasoning agents.
            expert_reference: The "ground truth" or expert perspective to aim for.
            max_iterations: Maximum number of refinement iterations.
            model_client: Optional client for LLM calls in the generator.
        """
        self.logger = logging.getLogger(__name__)
        self.expert_reference = expert_reference
        self.max_iterations = max_iterations
        
        # 1. Initialize Components
        self.generator = AdversarialGenerator(model_client=model_client)
        
        # Initialize agents and their diffusion policies
        self.agent_map = {}
        self.diffusion_agents = {}
        for config in agent_configs:
            agent_id = config["id"]
            domain = config.get("domain", "general")
            
            # Create the reasoning agent
            agent = DomainReasoningAgent(domain=domain)
            self.agent_map[agent_id] = agent
            
            # Create a diffusion policy for this agent
            # Note: In a real system, these might be shared or specialized
            self.diffusion_agents[agent_id] = DiffusionPolicy()

        # 2. Setup Orchestration and Environment
        # OMAD Orchestrator handles coordination between diffusion policies
        self.orchestrator = OMADOrchestrator(
            agents=self.diffusion_agents,
            agent_metadata=agent_configs
        )
        
        # Environment manages the blackboard and agent interaction
        self.env = AgentEnvironment(orchestrator=self.orchestrator)
        for agent in self.agent_map.values():
            self.env.register_agent(agent)

        self.history = []

    def evaluate_performance(self, agent_responses: Dict[str, List[str]], expert_ref: str) -> float:
        """
        Evaluates how well the agent team performed against the expert reference.
        Returns a 'gap score' (lower is better, or higher is better depending on metric).
        For this mock, we'll return a simulated score representing consensus quality.
        """
        # Simulated metric: how many agents participated and how long their responses are
        total_len = sum(len(resp[-1]) for resp in agent_responses.values() if resp)
        # More content (simulating reasoning depth) reduces the "gap" in this mock
        gap_score = max(0.0, 1.0 - (total_len / 1000.0)) 
        return gap_score

    def run_iteration(self, initial_context: str) -> Dict[str, Any]:
        """
        Runs the full adversarial loop for a set number of iterations.
        """
        current_query = initial_context
        current_responses = {}
        
        iteration_results = []

        for i in range(self.max_iterations):
            self.logger.info(f"Starting Integrated Loop Iteration {i+1}/{self.max_iterations}")
            
            # a. Dispatch current question to the multi-agent environment
            # This uses OMAD under the hood via AgentEnvironment
            env_result = self.env.process_query(current_query)
            current_responses = env_result["responses"]
            
            # b. Collect coordinated reasoning responses (Summary for evaluation)
            # Use the blackboard summary as the primary output
            consensus_summary = self.env.get_blackboard_content()
            
            # c. Evaluate the gap-closing performance
            gap_score = self.evaluate_performance(current_responses, self.expert_reference)
            
            self.logger.info(f"Iteration {i+1} Gap Score: {gap_score:.4f}")
            
            # Record state
            result_entry = {
                "iteration": i,
                "query": current_query,
                "gap_score": gap_score,
                "consensus_summary": consensus_summary
            }
            iteration_results.append(result_entry)
            
            # d. Update/Consult the generator with the results for the next iteration
            # Generates a more challenging question based on the current consensus
            current_query = self.generator.generate_question(
                original_prompt=current_query,
                target_response=consensus_summary,
                expert_reference=self.expert_reference
            )

        self.history = iteration_results
        return {
            "final_query": current_query,
            "final_gap_score": iteration_results[-1]["gap_score"],
            "history": self.history
        }

if __name__ == "__main__":
    # Basic smoke test configuration
    logging.basicConfig(level=logging.INFO)
    configs = [
        {"id": "agent_1", "domain": "physics", "morphology": {"expertise": "science"}},
        {"id": "agent_2", "domain": "math", "morphology": {"expertise": "science"}},
        {"id": "agent_3", "domain": "ethics", "morphology": {"expertise": "humanities"}},
    ]
    
    expert_text = "The unified theory must account for both quantum effects and gravitational constants."
    
    loop = IntegratedAdversarialLoop(configs, expert_text)
    final_state = loop.run_iteration("How do we unify physics?")
    print(f"Loop completed with final gap score: {final_state['final_gap_score']}")
