import pytest
import os
import plotly.graph_objects as go
from src.payload import ReasoningPayload
from src.tracker import LiveDriftTracker
from src.sensing import UncertaintySenser
from src.drift import DriftCalculator
from src.regulating import TruthRegulator
from src.display import DriftPlotter

def setup_plotter_suite():
    tracker = LiveDriftTracker()
    drift_calc = DriftCalculator(tracker)
    senser = UncertaintySenser(tracker)
    regulator = TruthRegulator(tracker, senser, drift_calc)
    plotter = DriftPlotter(tracker, regulator)
    return tracker, regulator, plotter

def test_drift_plotter_init():
    tracker, regulator, plotter = setup_plotter_suite()
    assert plotter.tracker == tracker
    assert plotter.regulator == regulator

def test_create_dashboard_empty():
    tracker, regulator, plotter = setup_plotter_suite()
    fig = plotter.create_dashboard()
    assert isinstance(fig, go.Figure)
    # Check if there are no nodes/edges in traces
    assert len(fig.data[1].x) == 0

def test_create_dashboard_with_data():
    tracker, regulator, plotter = setup_plotter_suite()
    
    # Add some nodes
    p1 = ReasoningPayload(source_id="n1", content="Root Intent", drift_score=0.0)
    p2 = ReasoningPayload(source_id="n2", content="Branch 1", drift_score=0.2)
    p3 = ReasoningPayload(source_id="n3", content="Drifting Branch 2", drift_score=0.8)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    tracker.add_reasoning_node(p3, parent_ids=["n1"])
    
    fig = plotter.create_dashboard()
    assert isinstance(fig, go.Figure)
    
    # 3 nodes
    assert len(fig.data[1].x) == 3
    # 2 edges (2 * 3 entries per edge in Scatter line: [x0, x1, None])
    assert len(fig.data[0].x) == 6

def test_save_dashboard():
    tracker, regulator, plotter = setup_plotter_suite()
    
    p1 = ReasoningPayload(source_id="n1", content="Root Intent", drift_score=0.0)
    tracker.add_reasoning_node(p1)
    
    test_file = "visuals/test_drift_dashboard.html"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    plotter.save_dashboard(test_file)
    
    assert os.path.exists(test_file)
    assert os.path.getsize(test_file) > 0
    
    # Cleanup
    os.remove(test_file)

def test_generate_report_summary():
    tracker, regulator, plotter = setup_plotter_suite()
    
    p1 = ReasoningPayload(source_id="n1", content="Root Intent", drift_score=0.0)
    p2 = ReasoningPayload(source_id="n2", content="Drifting bad.", drift_score=0.9)
    
    tracker.add_reasoning_node(p1)
    tracker.add_reasoning_node(p2, parent_ids=["n1"])
    
    summary = plotter.generate_report_summary()
    assert "# CAD-TRACE Drift Summary" in summary
    assert "Total Nodes:** 2" in summary
    assert "Status:** DRIFTING" in summary
    assert "Drift Origin:** n2" in summary
    assert "[n2]" in summary
