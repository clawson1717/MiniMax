import unittest
from src.trajectory import TrajectoryGraph

class TestTrajectoryGraph(unittest.TestCase):
    def setUp(self):
        self.graph = TrajectoryGraph()

    def test_add_step_single(self):
        state1 = "state1"
        action = "click"
        state2 = "state2"
        self.graph.add_step(state1, action, state2)
        
        path = self.graph.get_path()
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], (state1, action, state2))
        
        # Check node structure
        self.assertIn(state1, self.graph.nodes)
        self.assertIn(state2, self.graph.nodes)
        self.assertEqual(self.graph.nodes[state1], [(action, state2)])

    def test_add_step_cycle(self):
        s1, s2 = "s1", "s2"
        a1, a2 = "a1", "a2"
        
        # S1 --a1--> S2 --a2--> S1
        self.graph.add_step(s1, a1, s2)
        self.graph.add_step(s2, a2, s1)
        
        path = self.graph.get_path()
        self.assertEqual(len(path), 2)
        self.assertEqual(path, [(s1, a1, s2), (s2, a2, s1)])
        
        self.assertEqual(self.graph.nodes[s1], [(a1, s2)])
        self.assertEqual(self.graph.nodes[s2], [(a2, s1)])

    def test_repeated_state_and_action(self):
        s1, s2 = "s1", "s2"
        a1 = "a1"
        
        # S1 --a1--> S2, then S1 --a1--> S2 again
        self.graph.add_step(s1, a1, s2)
        self.graph.add_step(s1, a1, s2)
        
        path = self.graph.get_path()
        self.assertEqual(len(path), 2)
        
        # Nodes/edges should not duplicate the transition if identical
        self.assertEqual(len(self.graph.nodes[s1]), 1)
        self.assertEqual(self.graph.nodes[s1], [(a1, s2)])

    def test_reset(self):
        self.graph.add_step("s1", "a1", "s2")
        self.graph.reset()
        
        self.assertEqual(len(self.graph.get_path()), 0)
        self.assertEqual(len(self.graph.nodes), 0)

if __name__ == "__main__":
    unittest.main()
