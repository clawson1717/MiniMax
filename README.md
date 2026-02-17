# Pruned Adaptive Agent

An adaptive web agent combining CATTS, WebClipper, and CM2 approaches with:
- Uncertainty quantification
- Trajectory graph with pruning
- Checklist-based task tracking
- Recovery mechanisms

## Architecture

### Core Components

- **AdaptiveWebAgent**: Main agent orchestrator
- **UncertaintyEstimator**: Quantifies action uncertainty
- **TrajectoryGraph**: Graph-based trajectory management
- **Checklist**: Task dependency tracking
- **RecoveryManager**: Failure recovery strategies

## Installation

```bash
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
