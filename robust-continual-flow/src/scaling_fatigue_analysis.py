"""
Scaling vs. Fatigue Analysis

Analyzes the relationship between test-time scaling (compute allocation) and
the onset of "Reasoning Fatigue" in long-duration agent sessions.

Based on research from:
- CATTS (Lee et al., 2026): Test-time scaling strategies
- Consistency of LRMs under Multi-Turn Attacks (Li et al., 2026): Fatigue metrics
"""

import json
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import math


@dataclass
class SessionMetrics:
    """Metrics for a single session."""
    session_id: str
    duration_seconds: float
    total_steps: int
    total_tokens: int
    avg_compute_per_step: float
    fatigue_score: float
    fatigue_onset_step: Optional[int]
    scaling_strategy: str
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FatigueAnalysisResult:
    """Result of fatigue analysis."""
    session_id: str
    fatigue_detected: bool
    fatigue_onset_step: Optional[int]
    fatigue_onset_time: Optional[float]
    pre_fatigue_performance: float
    post_fatigue_performance: float
    performance_degradation: float
    compute_at_fatigue: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ScalingFatigueCorrelation:
    """Correlation between scaling and fatigue."""
    compute_level: str
    avg_fatigue_onset: float
    avg_performance: float
    sample_count: int
    correlation_coefficient: float


class ScalingFatigueAnalyzer:
    """
    Analyzes the relationship between test-time scaling and Reasoning Fatigue.
    
    Key research questions:
    1. Does higher compute allocation delay or accelerate fatigue onset?
    2. What's the optimal scaling strategy to maximize performance before fatigue?
    3. How does compute distribution over time affect fatigue?
    """
    
    def __init__(
        self,
        fatigue_threshold: float = 0.7,
        performance_window: int = 5,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the analyzer.
        
        Args:
            fatigue_threshold: Threshold for detecting fatigue onset
            performance_window: Window size for computing performance trends
            output_dir: Directory to save analysis results
        """
        self.fatigue_threshold = fatigue_threshold
        self.performance_window = performance_window
        self.output_dir = Path(output_dir) if output_dir else None
        self.sessions: List[SessionMetrics] = []
    
    def add_session(self, session: SessionMetrics) -> None:
        """Add a session to the analysis."""
        self.sessions.append(session)
    
    def analyze_session(self, session: SessionMetrics) -> FatigueAnalysisResult:
        """
        Analyze a single session for fatigue patterns.
        
        Args:
            session: Session metrics to analyze
            
        Returns:
            FatigueAnalysisResult with fatigue detection and recommendations
        """
        fatigue_detected = session.fatigue_score >= self.fatigue_threshold
        fatigue_onset = session.fatigue_onset_step
        
        # Estimate performance degradation
        if fatigue_detected and fatigue_onset:
            # Assume linear degradation for simplicity
            pre_fatigue = session.success_rate * 1.2  # Estimate
            post_fatigue = session.success_rate * 0.8  # Estimate
            degradation = (pre_fatigue - post_fatigue) / pre_fatigue if pre_fatigue > 0 else 0
        else:
            pre_fatigue = session.success_rate
            post_fatigue = session.success_rate
            degradation = 0.0
        
        # Compute at fatigue onset
        compute_at_fatigue = session.avg_compute_per_step if fatigue_onset else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            session, fatigue_detected, fatigue_onset
        )
        
        return FatigueAnalysisResult(
            session_id=session.session_id,
            fatigue_detected=fatigue_detected,
            fatigue_onset_step=fatigue_onset,
            fatigue_onset_time=fatigue_onset * session.duration_seconds / session.total_steps if fatigue_onset else None,
            pre_fatigue_performance=pre_fatigue,
            post_fatigue_performance=post_fatigue,
            performance_degradation=degradation,
            compute_at_fatigue=compute_at_fatigue,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        session: SessionMetrics,
        fatigue_detected: bool,
        fatigue_onset: Optional[int],
    ) -> List[str]:
        """Generate recommendations based on session analysis."""
        recommendations = []
        
        if not fatigue_detected:
            recommendations.append("No fatigue detected. Consider extending session duration.")
            recommendations.append(f"Current compute level ({session.avg_compute_per_step:.1f}) is sustainable.")
        else:
            if fatigue_onset and fatigue_onset < session.total_steps * 0.3:
                recommendations.append("Early fatigue onset detected. Consider reducing compute intensity.")
                recommendations.append("Implement periodic reset or context compression.")
            elif fatigue_onset and fatigue_onset > session.total_steps * 0.7:
                recommendations.append("Late fatigue onset. Session length is well-calibrated.")
            
            if session.avg_compute_per_step > 5:
                recommendations.append("High compute per step may accelerate fatigue. Consider adaptive scaling.")
            
            if session.scaling_strategy == "constant":
                recommendations.append("Consider dynamic scaling to manage fatigue.")
        
        return recommendations
    
    def analyze_scaling_correlation(self) -> List[ScalingFatigueCorrelation]:
        """
        Analyze correlation between compute scaling and fatigue onset.
        
        Returns:
            List of correlations for different compute levels
        """
        if not self.sessions:
            return []
        
        # Group sessions by compute level
        compute_levels = {
            "low": [],      # avg_compute < 3
            "medium": [],   # 3 <= avg_compute < 6
            "high": [],     # avg_compute >= 6
        }
        
        for session in self.sessions:
            if session.avg_compute_per_step < 3:
                compute_levels["low"].append(session)
            elif session.avg_compute_per_step < 6:
                compute_levels["medium"].append(session)
            else:
                compute_levels["high"].append(session)
        
        correlations = []
        for level, sessions in compute_levels.items():
            if not sessions:
                continue
            
            avg_fatigue_onset = sum(
                s.fatigue_onset_step or s.total_steps
                for s in sessions
            ) / len(sessions)
            
            avg_performance = sum(s.success_rate for s in sessions) / len(sessions)
            
            # Simple correlation: compute vs fatigue onset
            if len(sessions) > 1:
                compute_values = [s.avg_compute_per_step for s in sessions]
                fatigue_values = [s.fatigue_onset_step or s.total_steps for s in sessions]
                correlation = self._pearson_correlation(compute_values, fatigue_values)
            else:
                correlation = 0.0
            
            correlations.append(ScalingFatigueCorrelation(
                compute_level=level,
                avg_fatigue_onset=avg_fatigue_onset,
                avg_performance=avg_performance,
                sample_count=len(sessions),
                correlation_coefficient=correlation,
            ))
        
        return correlations
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n != len(y) or n == 0:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Dictionary with full analysis results
        """
        session_analyses = [self.analyze_session(s) for s in self.sessions]
        correlations = self.analyze_scaling_correlation()
        
        # Summary statistics
        total_sessions = len(self.sessions)
        fatigued_sessions = sum(1 for a in session_analyses if a.fatigue_detected)
        avg_fatigue_onset = None
        onset_values = [a.fatigue_onset_step for a in session_analyses if a.fatigue_onset_step]
        if onset_values:
            avg_fatigue_onset = sum(onset_values) / len(onset_values)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_sessions": total_sessions,
                "fatigued_sessions": fatigued_sessions,
                "fatigue_rate": fatigued_sessions / total_sessions if total_sessions > 0 else 0,
                "avg_fatigue_onset_step": avg_fatigue_onset,
            },
            "correlations": [asdict(c) for c in correlations],
            "sessions": [asdict(a) for a in session_analyses],
            "recommendations": self._generate_global_recommendations(session_analyses),
        }
    
    def _generate_global_recommendations(
        self,
        analyses: List[FatigueAnalysisResult],
    ) -> List[str]:
        """Generate global recommendations across all sessions."""
        recommendations = []
        
        if not analyses:
            return ["No sessions analyzed. Add sessions to generate recommendations."]
        
        fatigue_rate = sum(1 for a in analyses if a.fatigue_detected) / len(analyses)
        
        if fatigue_rate > 0.5:
            recommendations.append("High fatigue rate detected. Review scaling strategies.")
        
        avg_degradation = sum(a.performance_degradation for a in analyses) / len(analyses)
        if avg_degradation > 0.2:
            recommendations.append(f"Significant performance degradation ({avg_degradation:.1%}). Consider earlier session termination.")
        
        # Check for optimal compute level
        correlations = self.analyze_scaling_correlation()
        if correlations:
            best = max(correlations, key=lambda c: c.avg_performance)
            recommendations.append(f"Best performance at {best.compute_level} compute level ({best.avg_performance:.2f} success rate).")
        
        return recommendations
    
    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save analysis report to file."""
        if not self.output_dir:
            self.output_dir = Path(".")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scaling_fatigue_analysis_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath


def run_analysis_simulation(
    num_sessions: int = 10,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a simulated analysis with synthetic sessions.
    
    Args:
        num_sessions: Number of simulated sessions
        output_dir: Directory for output
        
    Returns:
        Analysis report dictionary
    """
    import random
    
    analyzer = ScalingFatigueAnalyzer(output_dir=output_dir)
    
    scaling_strategies = ["constant", "linear_increase", "adaptive", "decay"]
    
    for i in range(num_sessions):
        # Simulate session metrics
        compute = random.uniform(2, 8)
        duration = random.uniform(60, 300)
        steps = random.randint(10, 50)
        
        # Higher compute tends to cause earlier fatigue
        base_fatigue_onset = 30 - compute * 2
        fatigue_onset = max(5, min(steps, int(base_fatigue_onset + random.gauss(0, 5))))
        
        # Success rate decreases with fatigue
        base_success = 0.8 - compute * 0.02 + random.gauss(0, 0.1)
        success_rate = max(0.3, min(1.0, base_success))
        
        fatigue_score = 0.5 + (steps - fatigue_onset) / steps * 0.4 if fatigue_onset < steps else 0.3
        
        session = SessionMetrics(
            session_id=f"sim_{i+1:03d}",
            duration_seconds=duration,
            total_steps=steps,
            total_tokens=int(compute * steps * 100),
            avg_compute_per_step=compute,
            fatigue_score=fatigue_score,
            fatigue_onset_step=fatigue_onset if fatigue_score >= 0.7 else None,
            scaling_strategy=random.choice(scaling_strategies),
            success_rate=success_rate,
        )
        
        analyzer.add_session(session)
    
    return analyzer.generate_report()


if __name__ == "__main__":
    # Run simulation
    report = run_analysis_simulation(num_sessions=15)
    print(f"Analysis Summary:")
    print(f"  Total sessions: {report['summary']['total_sessions']}")
    print(f"  Fatigue rate: {report['summary']['fatigue_rate']:.1%}")
    print(f"  Avg fatigue onset: step {report['summary']['avg_fatigue_onset_step']:.1f}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
