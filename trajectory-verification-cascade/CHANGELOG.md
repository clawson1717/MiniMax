# Changelog

All notable changes to the Trajectory Verification Cascade (TVC) project will be documented in this file.

## [1.0.0] - 2026-03-03

### Added
- **Step 1: Project Scaffold**: Initialized project structure, requirements, and basic configuration.
- **Step 2: Trajectory Node Model**: Implemented `TrajectoryNode` dataclass with status tracking and checklist integration.
- **Step 3: Trajectory Graph**: Developed `TrajectoryGraph` for managing directed reasoning paths and cycle detection.
- **Step 4: Checklist Verifier**: Created `ChecklistVerifier` for binary criteria evaluation and evidence collection.
- **Step 5: Failure Mode Detector**: Implemented adversarial detection for 5 patterns: Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, and Reasoning Fatigue.
- **Step 6: Cascade Engine**: Orchestrated the verification flow combining graph management, verification, and detection.
- **Step 7: Backtracking Strategy**: Added robust backtracking to find alternative reasoning paths when verification fails.
- **Step 8: Pruning Policy**: Implemented logic to eliminate cyclic or unproductive reasoning branches.
- **Step 9: Integration Layer**: Unified all components into the `TVCAgent` for end-to-end task execution.
- **Step 10: Benchmark Tasks**: Created a benchmark suite with adversarial test cases to measure system performance.
- **Step 11: CLI Interface**: Developed a comprehensive command-line tool `tvc-cli` for running tasks and visualizing graphs.
- **Step 12: Documentation**: Finalized project documentation, architecture diagrams, and usage guides.

### Features
- Graph-based reasoning trajectory management.
- Per-node binary checklist verification.
- Detection of 5 adversarial failure modes.
- Automated backtracking and branch pruning.
- ASCII and Mermaid visualization support.
- CLI and Python API access.

## [0.1.0] - Initial planning and conceptualization.
- Defined the "verification cascade" concept combining WebClipper, CM2, and Multi-Turn Attack Failure Modes.
