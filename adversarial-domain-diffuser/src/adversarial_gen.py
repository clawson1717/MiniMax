import logging

class AdversarialGenerator:
    """
    Generates challenging questions to highlight the semantic gap 
    between a target model and an expert model.
    """

    def __init__(self, model_client=None):
        """
        Initialize with an optional model client for LLM calls.
        """
        self.model_client = model_client
        self.logger = logging.getLogger(__name__)

    def generate_question(self, original_prompt: str, target_response: str, expert_reference: str) -> str:
        """
        Analyzes the target response against the expert reference and generates
         a new question that targets identified weaknesses.
        """
        if self.model_client:
            prompt = self._build_adversarial_prompt(original_prompt, target_response, expert_reference)
            response = self.model_client.generate(prompt)
            return response
        else:
            # Enhanced mock to simulate domain shift/discovery
            expert_keywords = [
                "immunogenicity", "phenotypic", "off-target", "insertions", 
                "genomic", "modification", "stochastic", "viral vector",
                "nebula", "uv", "t-tauri", "nucleosynthesis", "desorption",
                "stochastic", "vacuum", "silicate", "prebiotic"
            ]
            
            # Find what is already in the target response
            already_addressed = [w for w in expert_keywords if w.lower() in target_response.lower()]
            to_challenge = [w for w in expert_keywords if w.lower() not in target_response.lower()]
            
            if to_challenge:
                # Injection of NEW adversarial concepts from the expert reference
                next_word = to_challenge[0]
                return f"Wait, the previous analysis missed '{next_word}'. How does '{next_word}' impact the legal framework? (Derived from expert reference)"
            else:
                return f"Re-evaluating based on the complete expert perspective: {expert_reference[:50]}... (Refining based on expert reference)"

    def _build_adversarial_prompt(self, original_prompt: str, target_response: str, expert_reference: str) -> str:
        """
        Constructs the prompt for the adversarial question generation.
        """
        return f"""
        Original Prompt: {original_prompt}
        
        Target Model Response: {target_response}
        
        Expert Model Reference: {expert_reference}
        
        Task: Identify the semantic gaps, missing reasoning, or inaccuracies in the Target Model's response 
        compared to the Expert Model. Then, generate a follow-up "adversarial" question that would 
        force the model to confront these gaps directly.
        
        Adversarial Question:
        """
