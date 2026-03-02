import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict

class EmbodimentGrouper:
    """
    Groups specialized agents based on their 'cognitive morphology' 
    (domain expertise, reasoning style, or model type) to reduce gradient conflicts.
    """
    def __init__(self, agents: List[Dict[str, Any]]):
        """
        Initialize the EmbodimentGrouper.

        Args:
            agents: A list of dictionaries, where each dict contains at least:
                    - 'id': Unique identifier for the agent
                    - 'morphology': A dict or string describing the agent's characteristics
        """
        self.agents = agents
        self.groups = {}

    def _get_morphology_key(self, morphology: Any) -> str:
        """
        Helper to convert morphology metadata into a hashable grouping key.
        Simple heuristic: if it's a dict, use 'expertise' or 'style'; if string, use it directly.
        """
        if isinstance(morphology, str):
            return morphology
        if isinstance(morphology, dict):
            # Try to find a primary characteristic
            return morphology.get('expertise', morphology.get('style', morphology.get('type', 'generic')))
        return 'generic'

    def perform_grouping(self):
        """
        Groups agents based on their morphology characteristics.
        """
        grouped_dict = defaultdict(list)
        for agent in self.agents:
            morphology = agent.get('morphology', 'generic')
            key = self._get_morphology_key(morphology)
            grouped_dict[key].append(agent['id'])
        
        self.groups = dict(grouped_dict)

    def get_groups(self) -> Dict[str, List[str]]:
        """
        Returns the mapping of group keys to lists of agent IDs.
        """
        if not self.groups:
            self.perform_grouping()
        return self.groups
