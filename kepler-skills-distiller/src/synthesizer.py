import json
import os
import random
from typing import List, Dict, Any

class PedagogicalSynthesizer:
    """
    Synthesizes training data categorized by Bloom's Taxonomy tiers for scientific discovery tasks.
    """

    BLOOM_TIERS = [
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create"
    ]

    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_remember(self, num_examples: int) -> List[Dict[str, Any]]:
        templates = [
            {"q": "What is the standard unit of {quantity} in the SI system?", "a": "{unit}"},
            {"q": "Who is credited with the discovery of the {law}?", "a": "{scientist}"},
            {"q": "State the formula for {concept}.", "a": "{formula}"}
        ]
        data = [
            {"quantity": "force", "unit": "Newton (N)", "law": "Law of Universal Gravitation", "scientist": "Isaac Newton", "concept": "linear momentum", "formula": "p = mv"},
            {"quantity": "energy", "unit": "Joule (J)", "law": "Laws of Planetary Motion", "scientist": "Johannes Kepler", "concept": "kinetic energy", "formula": "KE = 1/2 mv^2"},
            {"quantity": "power", "unit": "Watt (W)", "law": "General Relativity", "scientist": "Albert Einstein", "concept": "Newton's Second Law", "formula": "F = ma"}
        ]
        
        examples = []
        for _ in range(num_examples):
            template = random.choice(templates)
            item = random.choice(data)
            examples.append({
                "tier": "Remember",
                "question": template["q"].format(**item),
                "answer": template["a"].format(**item)
            })
        return examples

    def generate_understand(self, num_examples: int) -> List[Dict[str, Any]]:
        templates = [
            {"q": "How does {var1} change if {var2} is {action} in the equation {formula}?", "a": "If {var2} is {action}, then {var1} will {result}."},
            {"q": "Explain the physical significance of the constant {constant} in {context}.", "a": "The constant {constant} represents {significance}."}
        ]
        data = [
            {"var1": "gravitational force", "var2": "the distance", "action": "doubled", "formula": "F = Gmm/r^2", "result": "decrease by a factor of 4", "constant": "G", "context": "gravitation", "significance": "the strength of gravity"},
            {"var1": "kinetic energy", "var2": "velocity", "action": "tripled", "formula": "KE = 1/2 mv^2", "result": "increase by a factor of 9", "constant": "c", "context": "special relativity", "significance": "the speed of light in a vacuum"}
        ]
        
        examples = []
        for _ in range(num_examples):
            template = random.choice(templates)
            item = random.choice(data)
            examples.append({
                "tier": "Understand",
                "question": template["q"].format(**item),
                "answer": template["a"].format(**item)
            })
        return examples

    def generate_apply(self, num_examples: int) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(num_examples):
            m1 = random.randint(1, 100)
            m2 = random.randint(1, 100)
            r = random.randint(1, 10)
            f = (m1 * m2) / (r**2) # Simplified G=1
            examples.append({
                "tier": "Apply",
                "question": f"Calculate the gravitational force (G=1) between two masses of {m1}kg and {m2}kg separated by {r}m.",
                "answer": f"{f:.2f} N"
            })
        return examples

    def generate_analyze(self, num_examples: int) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(num_examples):
            # Example: Kepler's 3rd Law data
            constant = random.uniform(0.5, 2.0)
            data_points = []
            for a in range(1, 6):
                t_sq = constant * (a**3)
                data_points.append(f"(a={a}, T^2={t_sq:.2f})")
            
            examples.append({
                "tier": "Analyze",
                "question": f"Given the following orbital data {', '.join(data_points)}, what is the relationship between semi-major axis 'a' and period 'T'?",
                "answer": "T^2 is proportional to a^3 (Kepler's Third Law)."
            })
        return examples

    def generate_evaluate(self, num_examples: int) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(num_examples):
            hypotheses = ["F = ma", "F = ma^2", "F = m/a"]
            correct = hypotheses[0]
            observation = "When mass is 2kg and acceleration is 3m/s^2, the measured force is 6N."
            
            examples.append({
                "tier": "Evaluate",
                "question": f"Hypotheses: {', '.join(hypotheses)}. Observation: {observation}. Which hypothesis is consistent with the data?",
                "answer": f"{correct}"
            })
        return examples

    def generate_create(self, num_examples: int) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(num_examples):
            examples.append({
                "tier": "Create",
                "question": "Propose a mathematical model for a force that increases linearly with distance from a central point and is proportional to the object's mass.",
                "answer": "F = k * m * r, where k is a constant, m is mass, and r is distance."
            })
        return examples

    def synthesize(self, num_examples_per_tier: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        full_dataset = {}
        all_examples = []
        
        generators = {
            "Remember": self.generate_remember,
            "Understand": self.generate_understand,
            "Apply": self.generate_apply,
            "Analyze": self.generate_analyze,
            "Evaluate": self.generate_evaluate,
            "Create": self.generate_create
        }
        
        for tier in self.BLOOM_TIERS:
            tier_examples = generators[tier](num_examples_per_tier)
            full_dataset[tier] = tier_examples
            all_examples.extend(tier_examples)
            
        # Save to files
        self.save_dataset(full_dataset, "bloom_split.json")
        self.save_jsonl(all_examples, "synthetic_training.jsonl")
        
        return full_dataset

    def save_dataset(self, data: Any, filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Dataset saved to {path}")

    def save_jsonl(self, data: List[Dict], filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Dataset saved to {path}")

if __name__ == "__main__":
    synthesizer = PedagogicalSynthesizer()
    synthesizer.synthesize(num_examples_per_tier=10)
