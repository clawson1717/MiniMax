import logging
from typing import List, Dict, Any, Optional
from src.reasoning_agent import DomainReasoningAgent
from src.omad import OMADOrchestrator

class AgentEnvironment:
    """
    A simulated environment for multiple reasoning agents to interact.
    Uses a blackboard architecture where agents can share their reasoning.
    Now supports OMAD coordination.
    """

    def __init__(self, orchestrator: Optional[OMADOrchestrator] = None):
        self.agents: List[DomainReasoningAgent] = []
        self.blackboard: List[Dict[str, Any]] = []
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)

    def set_orchestrator(self, orchestrator: OMADOrchestrator):
        """Sets the OMAD orchestrator for the environment."""
        self.orchestrator = orchestrator

    def register_agent(self, agent: DomainReasoningAgent):
        """Adds an agent to the environment."""
        self.agents.append(agent)
        self.logger.info(f"Registered agent for domain: {agent.domain}")

    def post_to_blackboard(self, agent_domain: str, content: str):
        """Allows agents to share reasoning steps."""
        entry = {
            "agent_domain": agent_domain,
            "content": content,
            "timestamp": len(self.blackboard)  # Simple sequence tracker
        }
        self.blackboard.append(entry)
        self.logger.debug(f"New entry on blackboard from {agent_domain}")

    def get_blackboard_content(self) -> str:
        """Returns a string representation of the shared blackboard."""
        return "\n\n".join([f"[{entry['agent_domain']}]: {entry['content']}" for entry in self.blackboard])

    def process_query(self, query: str, iterations: int = 1) -> Dict[str, Any]:
        """
        Runs a multi-agent reasoning cycle.
        Agents iteratively refine their answers based on shared information.
        If an orchestrator is present, it uses OMAD to coordinate trajectories.
        """
        self.blackboard = []  # Clear blackboard for a new query
        
        coordination_history = []

        for i in range(iterations):
            self.logger.info(f"Starting iteration {i+1}")
            
            # If we have an orchestrator, perform a coordination step
            if self.orchestrator:
                coord_result = self.orchestrator.step(query)
                coordination_history.append(coord_result)
                self.logger.info("OMAD coordination step completed.")

            for agent in self.agents:
                # In a real scenario, we might pass the blackboard content to the agent
                # For now, let's simulate the interaction by including blackboard in the query if not empty
                current_context = f"Query: {query}\n\nShared Blackboard:\n{self.get_blackboard_content()}"
                response = agent.generate_response(current_context)
                self.post_to_blackboard(agent.domain, response)

        # Basic consensus/summary - just taking the last entries for now
        results = {agent.domain: [] for agent in self.agents}
        for entry in self.blackboard:
            results[entry['agent_domain']].append(entry['content'])

        return {
            "query": query,
            "responses": results,
            "blackboard_history": self.blackboard,
            "coordination_history": coordination_history
        }
