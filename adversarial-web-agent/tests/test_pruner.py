import unittest

from src.trajectory import TrajectoryGraph
from src.pruner import (
    TrajectoryPruner,
    detect_cycles,
    detect_dead_branches,
    prune,
)


def _has_cycle(graph: TrajectoryGraph) -> bool:
    """Helper: True if the graph's node adjacency contains any simple cycle."""
    return len(detect_cycles(graph)) > 0


class TestDetectCycles(unittest.TestCase):
    def test_detect_cycles_simple(self):
        g = TrajectoryGraph()
        # A -click-> B -back-> A
        g.add_step("A", "click", "B")
        g.add_step("B", "back", "A")

        cycles = detect_cycles(g)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0]), {"A", "B"})

    def test_detect_cycles_none(self):
        g = TrajectoryGraph()
        # Linear: A -> B -> C -> D
        g.add_step("A", "a1", "B")
        g.add_step("B", "a2", "C")
        g.add_step("C", "a3", "D")

        cycles = detect_cycles(g)
        self.assertEqual(cycles, [])

    def test_detect_cycles_multiple(self):
        g = TrajectoryGraph()
        # Cycle 1: A <-> B
        g.add_step("A", "x", "B")
        g.add_step("B", "y", "A")
        # Cycle 2: C -> D -> E -> C
        g.add_step("C", "p", "D")
        g.add_step("D", "q", "E")
        g.add_step("E", "r", "C")

        cycles = detect_cycles(g)
        self.assertEqual(len(cycles), 2)
        cycle_sets = [set(c) for c in cycles]
        self.assertIn({"A", "B"}, cycle_sets)
        self.assertIn({"C", "D", "E"}, cycle_sets)


class TestDetectDeadBranches(unittest.TestCase):
    def test_detect_dead_branches(self):
        g = TrajectoryGraph()
        # Main path reaches the goal:  START -> HOME -> GOAL
        g.add_step("START", "nav", "HOME")
        g.add_step("HOME", "submit", "GOAL")
        # Dead branch off HOME that never returns to anything reaching GOAL.
        g.add_step("HOME", "detour", "DEAD1")
        g.add_step("DEAD1", "next", "DEAD2")

        dead = detect_dead_branches(g, goal_states={"GOAL"})
        self.assertEqual(set(dead), {"DEAD1", "DEAD2"})

    def test_detect_dead_branches_all_reachable(self):
        g = TrajectoryGraph()
        g.add_step("A", "a1", "B")
        g.add_step("B", "a2", "C")
        g.add_step("A", "a3", "C")  # A also reaches C directly

        dead = detect_dead_branches(g, goal_states={"C"})
        self.assertEqual(dead, [])


class TestPrune(unittest.TestCase):
    def test_prune_cycles(self):
        g = TrajectoryGraph()
        # A -> B -> A -> B -> C  (traversal that loops once)
        g.add_step("A", "go", "B")
        g.add_step("B", "back", "A")
        g.add_step("A", "go", "B")
        g.add_step("B", "forward", "C")

        self.assertTrue(_has_cycle(g))
        original_traj_len = len(g.trajectory)

        pruned = prune(g)

        # No cycles remain.
        self.assertFalse(_has_cycle(pruned))
        # Trajectory became shorter (the back-edge step dropped).
        self.assertLess(len(pruned.trajectory), original_traj_len)
        # Forward progress is preserved.
        self.assertIn(("B", "forward", "C"), pruned.trajectory)
        # Input graph was not mutated.
        self.assertTrue(_has_cycle(g))
        self.assertEqual(len(g.trajectory), original_traj_len)

    def test_prune_dead_branches(self):
        g = TrajectoryGraph()
        g.add_step("START", "nav", "HOME")
        g.add_step("HOME", "submit", "GOAL")
        g.add_step("HOME", "detour", "DEAD1")
        g.add_step("DEAD1", "next", "DEAD2")

        pruned = prune(g, goal_states={"GOAL"})

        # Dead nodes gone from the adjacency map.
        self.assertNotIn("DEAD1", pruned.nodes)
        self.assertNotIn("DEAD2", pruned.nodes)
        # Survivor nodes preserved.
        self.assertIn("START", pruned.nodes)
        self.assertIn("HOME", pruned.nodes)
        self.assertIn("GOAL", pruned.nodes)
        # The edge leading into the dead branch is gone from HOME's neighbours.
        home_targets = {nxt for (_a, nxt) in pruned.nodes["HOME"]}
        self.assertNotIn("DEAD1", home_targets)
        # Trajectory steps touching dead nodes were removed.
        for step in pruned.trajectory:
            self.assertNotIn("DEAD1", (step[0], step[2]))
            self.assertNotIn("DEAD2", (step[0], step[2]))
        # Main path step is preserved.
        self.assertIn(("HOME", "submit", "GOAL"), pruned.trajectory)
        # Input graph was not mutated.
        self.assertIn("DEAD1", g.nodes)
        self.assertIn("DEAD2", g.nodes)

    def test_prune_no_goal_states(self):
        g = TrajectoryGraph()
        # Cycle between A and B, plus an unrelated dead branch D -> E.
        g.add_step("A", "go", "B")
        g.add_step("B", "loop", "A")
        g.add_step("D", "hop", "E")

        pruned_none = prune(g, goal_states=None)
        pruned_empty = prune(g, goal_states=set())

        # Cycles removed...
        self.assertFalse(_has_cycle(pruned_none))
        self.assertFalse(_has_cycle(pruned_empty))
        # ...but dead branches (D, E) are kept because we had no goal.
        self.assertIn("D", pruned_none.nodes)
        self.assertIn("E", pruned_none.nodes)
        self.assertIn("D", pruned_empty.nodes)
        self.assertIn("E", pruned_empty.nodes)


class TestTrajectoryPrunerClass(unittest.TestCase):
    def test_class_delegates_to_module_functions(self):
        g = TrajectoryGraph()
        g.add_step("A", "a1", "B")
        g.add_step("B", "a2", "A")
        g.add_step("B", "a3", "C")

        p = TrajectoryPruner()
        self.assertEqual(len(p.detect_cycles(g)), 1)
        self.assertEqual(p.detect_dead_branches(g, {"C"}), [])

        pruned = p.prune(g, goal_states={"C"})
        self.assertFalse(_has_cycle(pruned))
        self.assertIn("C", pruned.nodes)


if __name__ == "__main__":
    unittest.main()
