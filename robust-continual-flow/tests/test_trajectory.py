import unittest
import os
import json
from src.trajectory import TrajectoryLogger

class TestTrajectoryLogger(unittest.TestCase):
    def setUp(self):
        self.logger = TrajectoryLogger()

    def test_add_node(self):
        node_id = self.logger.add_node("state 1", {"uncertainty": 0.1})
        self.assertIn(node_id, self.logger.nodes)
        self.assertEqual(self.logger.nodes[node_id]["observation"], "state 1")
        self.assertEqual(self.logger.nodes[node_id]["metadata"]["uncertainty"], 0.1)

    def test_add_edge(self):
        n1 = self.logger.add_node("obs 1")
        n2 = self.logger.add_node("obs 2")
        self.logger.add_edge(n1, n2, "action A", {"cost": 0.5})
        
        self.assertEqual(len(self.logger.edges), 1)
        edge = self.logger.edges[0]
        self.assertEqual(edge["from"], n1)
        self.assertEqual(edge["to"], n2)
        self.assertEqual(edge["action"], "action A")
        self.assertEqual(edge["metadata"]["cost"], 0.5)

    def test_invalid_edge(self):
        n1 = self.logger.add_node("obs 1")
        with self.assertRaises(ValueError):
            self.logger.add_edge(n1, "non-existent", "action")

    def test_serialization(self):
        n1 = self.logger.add_node("obs 1")
        n2 = self.logger.add_node("obs 2")
        self.logger.add_edge(n1, n2, "action A")
        
        filepath = "test_graph.json"
        try:
            self.logger.save(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            loaded_logger = TrajectoryLogger.load(filepath)
            self.assertEqual(len(loaded_logger.nodes), 2)
            self.assertEqual(len(loaded_logger.edges), 1)
            self.assertEqual(loaded_logger.edges[0]["action"], "action A")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == "__main__":
    unittest.main()
