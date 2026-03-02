"""
Tests for Scaling vs. Fatigue Analysis.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.scaling_fatigue_analysis import (
    SessionMetrics,
    FatigueAnalysisResult,
    ScalingFatigueCorrelation,
    ScalingFatigueAnalyzer,
    run_analysis_simulation,
)


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""
    
    def test_creation(self):
        """Test creating session metrics."""
        session = SessionMetrics(
            session_id="test_001",
            duration_seconds=120.0,
            total_steps=20,
            total_tokens=5000,
            avg_compute_per_step=4.5,
            fatigue_score=0.6,
            fatigue_onset_step=15,
            scaling_strategy="adaptive",
            success_rate=0.75,
        )
        
        assert session.session_id == "test_001"
        assert session.duration_seconds == 120.0
        assert session.total_steps == 20
        assert session.avg_compute_per_step == 4.5
        assert session.fatigue_score == 0.6
    
    def test_default_metadata(self):
        """Test default metadata is empty dict."""
        session = SessionMetrics(
            session_id="test",
            duration_seconds=60,
            total_steps=10,
            total_tokens=1000,
            avg_compute_per_step=3.0,
            fatigue_score=0.5,
            fatigue_onset_step=None,
            scaling_strategy="constant",
            success_rate=0.8,
        )
        
        assert session.metadata == {}


class TestFatigueAnalysisResult:
    """Tests for FatigueAnalysisResult dataclass."""
    
    def test_creation(self):
        """Test creating analysis result."""
        result = FatigueAnalysisResult(
            session_id="test_001",
            fatigue_detected=True,
            fatigue_onset_step=15,
            fatigue_onset_time=90.0,
            pre_fatigue_performance=0.85,
            post_fatigue_performance=0.65,
            performance_degradation=0.24,
            compute_at_fatigue=4.5,
        )
        
        assert result.fatigue_detected is True
        assert result.fatigue_onset_step == 15
        assert result.performance_degradation == 0.24
    
    def test_default_recommendations(self):
        """Test default recommendations is empty list."""
        result = FatigueAnalysisResult(
            session_id="test",
            fatigue_detected=False,
            fatigue_onset_step=None,
            fatigue_onset_time=None,
            pre_fatigue_performance=0.8,
            post_fatigue_performance=0.8,
            performance_degradation=0.0,
            compute_at_fatigue=0.0,
        )
        
        assert result.recommendations == []


class TestScalingFatigueAnalyzer:
    """Tests for ScalingFatigueAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ScalingFatigueAnalyzer()
        
        assert analyzer.fatigue_threshold == 0.7
        assert analyzer.performance_window == 5
        assert analyzer.sessions == []
    
    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        analyzer = ScalingFatigueAnalyzer(
            fatigue_threshold=0.8,
            performance_window=10,
            output_dir="/tmp/analysis",
        )
        
        assert analyzer.fatigue_threshold == 0.8
        assert analyzer.performance_window == 10
    
    def test_add_session(self):
        """Test adding sessions."""
        analyzer = ScalingFatigueAnalyzer()
        session = SessionMetrics(
            session_id="test",
            duration_seconds=60,
            total_steps=10,
            total_tokens=1000,
            avg_compute_per_step=3.0,
            fatigue_score=0.5,
            fatigue_onset_step=None,
            scaling_strategy="constant",
            success_rate=0.8,
        )
        
        analyzer.add_session(session)
        
        assert len(analyzer.sessions) == 1
    
    def test_analyze_session_no_fatigue(self):
        """Test analyzing session without fatigue."""
        analyzer = ScalingFatigueAnalyzer()
        session = SessionMetrics(
            session_id="healthy_session",
            duration_seconds=120,
            total_steps=20,
            total_tokens=3000,
            avg_compute_per_step=3.0,
            fatigue_score=0.3,  # Below threshold
            fatigue_onset_step=None,
            scaling_strategy="constant",
            success_rate=0.85,
        )
        
        result = analyzer.analyze_session(session)
        
        assert result.fatigue_detected is False
        assert result.fatigue_onset_step is None
        assert "No fatigue detected" in result.recommendations[0]
    
    def test_analyze_session_with_fatigue(self):
        """Test analyzing session with fatigue."""
        analyzer = ScalingFatigueAnalyzer()
        session = SessionMetrics(
            session_id="fatigued_session",
            duration_seconds=180,
            total_steps=30,
            total_tokens=6000,
            avg_compute_per_step=4.0,
            fatigue_score=0.8,  # Above threshold
            fatigue_onset_step=20,
            scaling_strategy="constant",
            success_rate=0.6,
        )
        
        result = analyzer.analyze_session(session)
        
        assert result.fatigue_detected is True
        assert result.fatigue_onset_step == 20
        assert len(result.recommendations) > 0
    
    def test_analyze_session_early_fatigue(self):
        """Test detecting early fatigue onset."""
        analyzer = ScalingFatigueAnalyzer()
        session = SessionMetrics(
            session_id="early_fatigue",
            duration_seconds=120,
            total_steps=30,
            total_tokens=5000,
            avg_compute_per_step=6.0,
            fatigue_score=0.85,
            fatigue_onset_step=5,  # Early (less than 30% of steps)
            scaling_strategy="constant",
            success_rate=0.5,
        )
        
        result = analyzer.analyze_session(session)
        
        assert result.fatigue_detected is True
        assert any("Early fatigue onset" in r for r in result.recommendations)
    
    def test_analyze_scaling_correlation_empty(self):
        """Test correlation analysis with no sessions."""
        analyzer = ScalingFatigueAnalyzer()
        
        correlations = analyzer.analyze_scaling_correlation()
        
        assert correlations == []
    
    def test_analyze_scaling_correlation(self):
        """Test correlation analysis with sessions."""
        analyzer = ScalingFatigueAnalyzer()
        
        # Add sessions with different compute levels
        for i, compute in enumerate([2.0, 2.5, 4.0, 4.5, 7.0, 7.5]):
            session = SessionMetrics(
                session_id=f"session_{i}",
                duration_seconds=100,
                total_steps=20,
                total_tokens=int(compute * 2000),
                avg_compute_per_step=compute,
                fatigue_score=0.5,
                fatigue_onset_step=15 if compute < 4 else 10,
                scaling_strategy="constant",
                success_rate=0.8 - compute * 0.05,
            )
            analyzer.add_session(session)
        
        correlations = analyzer.analyze_scaling_correlation()
        
        assert len(correlations) == 3  # low, medium, high
        levels = {c.compute_level for c in correlations}
        assert "low" in levels
        assert "medium" in levels
        assert "high" in levels
    
    def test_generate_report(self):
        """Test report generation."""
        analyzer = ScalingFatigueAnalyzer()
        
        for i in range(5):
            session = SessionMetrics(
                session_id=f"session_{i}",
                duration_seconds=100,
                total_steps=20,
                total_tokens=3000,
                avg_compute_per_step=4.0,
                fatigue_score=0.6 if i < 3 else 0.8,
                fatigue_onset_step=15 if i >= 3 else None,
                scaling_strategy="constant",
                success_rate=0.75,
            )
            analyzer.add_session(session)
        
        report = analyzer.generate_report()
        
        assert "timestamp" in report
        assert "summary" in report
        assert report["summary"]["total_sessions"] == 5
        assert "correlations" in report
        assert "recommendations" in report
    
    def test_save_report(self):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ScalingFatigueAnalyzer(output_dir=tmpdir)
            
            session = SessionMetrics(
                session_id="test",
                duration_seconds=60,
                total_steps=10,
                total_tokens=1000,
                avg_compute_per_step=3.0,
                fatigue_score=0.5,
                fatigue_onset_step=None,
                scaling_strategy="constant",
                success_rate=0.8,
            )
            analyzer.add_session(session)
            
            filepath = analyzer.save_report()
            
            assert filepath.exists()
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "summary" in data


class TestRunAnalysisSimulation:
    """Tests for simulation function."""
    
    def test_simulation_returns_report(self):
        """Test that simulation returns valid report."""
        report = run_analysis_simulation(num_sessions=5)
        
        assert "summary" in report
        assert report["summary"]["total_sessions"] == 5
        assert "correlations" in report
        assert "recommendations" in report
    
    def test_simulation_with_output_dir(self):
        """Test simulation with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_analysis_simulation(num_sessions=3, output_dir=tmpdir)
            
            assert report is not None
    
    def test_simulation_fatigue_distribution(self):
        """Test that simulation produces varied fatigue results."""
        report = run_analysis_simulation(num_sessions=20)
        
        # With 20 sessions, we should see some fatigue
        assert report["summary"]["fatigue_rate"] > 0


class TestPearsonCorrelation:
    """Tests for correlation calculation."""
    
    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation."""
        analyzer = ScalingFatigueAnalyzer()
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        
        corr = analyzer._pearson_correlation(x, y)
        
        assert abs(corr - 1.0) < 0.01
    
    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        analyzer = ScalingFatigueAnalyzer()
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        
        corr = analyzer._pearson_correlation(x, y)
        
        assert abs(corr + 1.0) < 0.01
    
    def test_no_correlation(self):
        """Test zero correlation."""
        analyzer = ScalingFatigueAnalyzer()
        x = [1, 2, 3, 4, 5]
        y = [5, 5, 5, 5, 5]  # Constant
        
        corr = analyzer._pearson_correlation(x, y)
        
        assert corr == 0.0
    
    def test_empty_lists(self):
        """Test with empty lists."""
        analyzer = ScalingFatigueAnalyzer()
        
        corr = analyzer._pearson_correlation([], [])
        
        assert corr == 0.0


class TestIntegration:
    """Integration tests for the full analysis pipeline."""
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ScalingFatigueAnalyzer(output_dir=tmpdir)
            
            # Simulate a realistic mix of sessions
            sessions = [
                SessionMetrics("s1", 120, 20, 4000, 2.5, 0.4, None, "constant", 0.85),
                SessionMetrics("s2", 180, 30, 9000, 5.0, 0.75, 20, "adaptive", 0.70),
                SessionMetrics("s3", 90, 15, 2000, 2.0, 0.3, None, "constant", 0.90),
                SessionMetrics("s4", 240, 40, 12000, 6.0, 0.85, 15, "linear_increase", 0.55),
                SessionMetrics("s5", 150, 25, 5000, 3.5, 0.5, None, "decay", 0.80),
            ]
            
            for session in sessions:
                analyzer.add_session(session)
            
            # Generate and save report
            filepath = analyzer.save_report("test_analysis.json")
            
            assert filepath.exists()
            
            with open(filepath, 'r') as f:
                report = json.load(f)
            
            # Verify report structure
            assert report["summary"]["total_sessions"] == 5
            assert len(report["sessions"]) == 5
            assert len(report["recommendations"]) > 0
