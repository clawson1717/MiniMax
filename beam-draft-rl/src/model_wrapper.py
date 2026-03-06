import re

class DraftWrapper:
    """
    Wrapper for handling "draft" reasoning tokens and sections.
    """
    DRAFT_START = "<|draft_start|>"
    DRAFT_END = "<|draft_end|>"

    def __init__(self, model_name: str = None):
        self.model_name = model_name

    def wrap_prompt(self, prompt: str) -> str:
        """
        Wraps the prompt to instruct the model to use draft tokens.
        """
        instruction = f"\nPlease use {self.DRAFT_START} and {self.DRAFT_END} to enclose your internal draft reasoning before providing the final answer."
        return prompt + instruction

    def extract_draft(self, output: str) -> str:
        """
        Extracts the draft content from the model output.
        """
        pattern = f"{re.escape(self.DRAFT_START)}(.*?){re.escape(self.DRAFT_END)}"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def compress_cot(self, cot_text: str) -> str:
        """
        Placeholder logic to shorten CoT text. 
        In the future, this could use a smaller model or rule-based summarization.
        """
        # Simple placeholder: return first 100 chars if longer
        if len(cot_text) > 100:
            return cot_text[:97] + "..."
        return cot_text
