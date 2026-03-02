import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ReasoningAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device_map="auto"):
        self.model_id = model_id
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )

    def generate_reasoning(self, prompt: str, num_samples: int = 1, temperature: float = 0.7) -> list[str]:
        system_prompt = "You are a helpful assistant that thinks step-by-step before answering."
        user_prompt = f"Problem: {prompt}\n\nThink step by step and then provide the final answer."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=num_samples,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the generated part
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = [
            output_ids[input_length:] for output_ids in generated_ids
        ]
        
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]

    def refine_reasoning(self, prompt: str, previous_reasoning: str, temperature: float = 0.5) -> str:
        system_prompt = "You are a helpful assistant that refines and improves reasoning traces."
        user_prompt = (
            f"Problem: {prompt}\n\n"
            f"Previous Reasoning Trace:\n{previous_reasoning}\n\n"
            "Analyze the previous reasoning, identify any errors or areas for improvement, "
            "and provide a refined, more accurate, and clearer reasoning trace with the final answer."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = [
            output_ids[input_length:] for output_ids in generated_ids
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def get_logits(self, prompt: str) -> torch.Tensor:
        """
        Returns the logits for the last token generated for a given prompt.
        Used for information-theoretic evaluations.
        """
        system_prompt = "You are a helpful assistant that thinks step-by-step before answering."
        user_prompt = f"Problem: {prompt}\n\nThink step by step and then provide the final answer."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            # Get logits for the last position
            logits = outputs.logits[:, -1, :]
            
        return logits
