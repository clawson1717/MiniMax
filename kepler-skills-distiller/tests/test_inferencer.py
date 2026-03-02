import unittest
import numpy as np
from src.inferencer import PhysicalPropertyInferencer

class TestPhysicalPropertyInferencer(unittest.TestCase):
    
    def test_circular_motion_conservation(self):
        # Create circular motion data: x^2 + y^2 = 1
        t = np.linspace(0, 10, 100)
        x = np.cos(t)
        y = np.sin(t)
        
        data = {"t": t, "x": x, "y": y}
        inferencer = PhysicalPropertyInferencer(data)
        constraints = inferencer.infer_all()
        
        # Verify conservation laws detected the sum of squares
        laws = constraints["conservation_laws"]
        found = any(law["expression"] == "x^2 + y^2" for law in laws)
        self.assertTrue(found, "Should have detected x^2 + y^2 conservation")

    def test_time_symmetry(self):
        # Create stationary data
        t = np.linspace(0, 10, 100)
        v = np.ones(100) * 5.0 # Constant velocity
        
        data = {"t": t, "v": v}
        inferencer = PhysicalPropertyInferencer(data)
        constraints = inferencer.infer_all()
        
        # Should detect time-translation symmetry for v
        symmetries = constraints["symmetries"]
        found = any(s["type"] == "time-translation" and s["variable"] == "v" for s in symmetries)
        self.assertTrue(found, "Should have detected time-translation symmetry for v")

if __name__ == "__main__":
    unittest.main()
