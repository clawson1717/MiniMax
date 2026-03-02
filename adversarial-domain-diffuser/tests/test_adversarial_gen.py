import unittest
from src.adversarial_gen import AdversarialGenerator

class MockModelClient:
    def generate(self, prompt: str) -> str:
        return "Mocked adversarial question based on gaps."

class TestAdversarialGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = AdversarialGenerator()
        self.generator_with_client = AdversarialGenerator(model_client=MockModelClient())

    def test_generate_question_mock(self):
        original = "What is gravity?"
        target = "Gravity is a force."
        expert = "Gravity is a curvature of spacetime described by General Relativity."
        
        question = self.generator.generate_question(original, target, expert)
        self.assertIsInstance(question, str)
        self.assertIn("expert reference", question.lower())

    def test_generate_question_with_client(self):
        original = "What is gravity?"
        target = "Gravity is a force."
        expert = "Gravity is a curvature of spacetime described by General Relativity."
        
        question = self.generator_with_client.generate_question(original, target, expert)
        self.assertEqual(question, "Mocked adversarial question based on gaps.")

    def test_build_prompt(self):
        original = "Prompt"
        target = "Target"
        expert = "Expert"
        prompt = self.generator._build_adversarial_prompt(original, target, expert)
        self.assertIn(original, prompt)
        self.assertIn(target, prompt)
        self.assertIn(expert, prompt)

if __name__ == "__main__":
    unittest.main()
