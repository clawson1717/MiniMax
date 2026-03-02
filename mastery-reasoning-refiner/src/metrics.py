import numpy as np
import torch
from typing import List, Union

class InformationTheoreticMetrics:
    """
    Implements information-theoretic metrics for reasoning agents, 
    inspired by Faustino (2026) regarding world model capacity.
    """

    @staticmethod
    def calculate_world_model_capacity(policy_probs: torch.Tensor, env_states: int) -> float:
        """
        Estimates the bits of information about the environment.
        As a baseline, uses I = n * log2(m) where n is the number of states
        and m is the action space size, but here we estimate from policy distribution.
        
        Args:
            policy_probs: Tensor of action probabilities (batch, num_actions)
            env_states: Number of discrete environment states considered.
            
        Returns:
            Estimated bits of information.
        """
        # Ensure probabilities sum to 1
        probs = torch.softmax(policy_probs, dim=-1) if not torch.allclose(policy_probs.sum(dim=-1), torch.tensor(1.0)) else policy_probs
        
        # Calculate entropy of the policy: H(π) = -Σ p(a|s) log2 p(a|s)
        # Higher entropy means less information (more uncertainty)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1)
        mean_entropy = torch.mean(entropy).item()
        
        # Max possible entropy for the action space
        num_actions = probs.shape[-1]
        max_entropy = np.log2(num_actions)
        
        # Information gain per state is the reduction from max uncertainty
        info_per_state = max_entropy - mean_entropy
        
        # Total capacity across all estimated environment states
        total_capacity = env_states * info_per_state
        
        return max(0.0, total_capacity)

    @staticmethod
    def calculate_semantic_information_gain(initial_trace: str, refined_trace: str) -> float:
        """
        Estimates 'semantic information gain' between an initial reasoning trace 
        and a refined one using entropy reduction proxy (token-level).
        
        Args:
            initial_trace: The raw first-pass reasoning text.
            refined_trace: The iteratively improved reasoning text.
            
        Returns:
            Bits of semantic information gained.
        """
        if not initial_trace or not refined_trace:
            return 0.0
            
        def get_char_probs(text: str):
            if not text: return {}
            counts = {}
            for char in text:
                counts[char] = counts.get(char, 0) + 1
            total = len(text)
            return {char: count/total for char, count in counts.items()}

        def calculate_entropy(probs: dict):
            return -sum(p * np.log2(p) for p in probs.values())

        p1 = get_char_probs(initial_trace)
        p2 = get_char_probs(refined_trace)
        
        h1 = calculate_entropy(p1)
        h2 = calculate_entropy(p2)
        
        # In this context, if the refined trace is more structured (lower entropy)
        # or provides more specific details, we consider the delta as gain.
        # However, a simple entropy delta doesn't capture 'meaning'.
        # For a text-based proxy, we look at the Kullback-Leibler Divergence
        # if we treat them as distributions, or simply the change in descriptive complexity.
        
        # Simplified Gain: Length-weighted entropy change
        # If refined is longer and has higher information density, it's a gain.
        gain = (len(refined_trace) * h2) - (len(initial_trace) * h1)
        
        # Normalize to bits per character relative to initial
        return gain / max(len(initial_trace), 1)
