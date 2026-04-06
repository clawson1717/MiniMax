# Project Usage Log: Kepler-Skills-Distiller

## Test Run [2026-03-04]

### Observations
- **Discovery Pipeline**: Successfully processed planetary orbit data (a/T relationship).
- **Inference**: Correctly identifies potential symmetries and conservation laws, though currently uses placeholders for some complex inferences.
- **Bloom's Taxonomy Distillation**: The curriculum progression works as expected in mock mode, transitioning from "Remember" up to "Create" based on mastery thresholds (85% accuracy).
- **Benchmark**: Successfully ran the Feynman Symbolic Regression benchmark. 

### Bugs & UX Friction
- **Slow Discovery**: The `discover` command has artificial `time.sleep` calls in `src/cli.py`, which helps simulate processing but might be annoying for power users if not configurable.
- **Mock Confusion**: The difference between "mock" mode and "real" mode in the CLI could be clearer (e.g., what happens when `--mock` is omitted without a configured model).
- **Physical Inference Output**: The `PhysicalPropertyInferencer` often returns "placeholder" or "unknown" for units when data doesn't have explicit metadata. Adding a way to specify units in the input JSON would improve this.
- **Benchmark Accuracy**: Mock models predictably show 0.0% accuracy on exact matches. For better "out of the box" demo experience, the mock expert could be improved to "discover" a few basic linear relationships.

### Ideas for New Features
- **Unit Metadata**: Support for `units` field in input data for better dimensional consistency checks.
- **Interactive Discovery**: A TUI or web interface to step through the "Think like a Scientist" reasoning iterations.
- **Skill Transfer Visualization**: A tool to visualize how a skill discovered in one domain (e.g., mechanics) is applied to another (e.g., electromagnetism) during distillation.
- **Live Training Plots**: Real-time plotting of loss and accuracy during the `distill` process.
