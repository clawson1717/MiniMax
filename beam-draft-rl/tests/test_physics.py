import unittest
from src.engine import PhysicsEngine
from src.reward import VerifiableReward

class TestPhysics(unittest.TestCase):
    def setUp(self):
        self.engine = PhysicsEngine()
        self.reward = VerifiableReward()

    def test_equilibrium(self):
        """
        Tests if a beam is in equilibrium.
        Length: 10m
        Loads: 100N at 5m
        Reactions: -50N at 0m, -50N at 10m
        """
        length = '10m'
        forces = [
            {'value': '100N', 'position': '5m'}
        ]
        reactions = [
            {'value': '-50N', 'position': '0m'},
            {'value': '-50N', 'position': '10m'}
        ]
        self.assertTrue(self.engine.validate_equilibrium(length, forces, reactions))

    def test_solve_simply_supported(self):
        """
        Tests calculating reactions for a simply supported beam.
        Length: 10m
        Loads: 100N at 5m
        Reactions: R0 = -50N, RL = -50N
        """
        results = self.engine.solve_simply_supported_beam('10m', '100N', '5m')
        self.assertAlmostEqual(results['R0'].magnitude, -50)
        self.assertAlmostEqual(results['RL'].magnitude, -50)

    def test_solve_point_load_offset(self):
        """
        Tests reactions for load at 2m on a 10m beam.
        RA = P * (L-a)/L = 100 * (10-2)/10 = 80N
        RB = P * a/L = 100 * 2/10 = 20N
        """
        results = self.engine.solve_simply_supported_beam('10m', '100N', '2m')
        self.assertAlmostEqual(results['R0'].magnitude, -80)
        self.assertAlmostEqual(results['RL'].magnitude, -20)

    def test_reward_binary(self):
        """
        Verifies simple binary reward on final answer.
        """
        reasoning = "The total force is 100N. The load is centered. Therefore, the reactions are 50N each.\nFinal Answer: 50N"
        ground_truth = "50N"
        reward_val = self.reward.calculate_reward(reasoning, ground_truth)
        self.assertEqual(reward_val, 1.0)
        
        reasoning2 = "Final Answer: 40N"
        reward_val2 = self.reward.calculate_reward(reasoning2, ground_truth)
        self.assertEqual(reward_val2, 0.0)

    def test_verify_equations(self):
        """
        Verifies simple equation logic.
        """
        eqs = ["50N + 50N - 100N = 0"]
        self.assertTrue(self.reward.verify_equations(eqs))

if __name__ == '__main__':
    unittest.main()
