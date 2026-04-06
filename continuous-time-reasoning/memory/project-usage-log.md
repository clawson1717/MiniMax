# CTRD Project Usage Log

## [2026-03-03] Early Implementation Usage Test (Steps 1-2)

### Summary of Actions
- Created `use_ctrd_state.py` to test importing and using `ReasoningState`.
- Simulated a trajectory of 5 `ReasoningState` objects with torch tensors.
- Attempted to serialize these states into standard Python dictionaries.

### Identified Friction / Gaps
- **Serialization Missing:** The `ReasoningState` class is a standard Python `dataclass` but lacks built-in serialization methods. Although `asdict(state)` works for simple dataclasses, `torch.Tensor` objects need special handling for JSON/Disk serialization. A `.to_dict()` or `model_dump()` approach should be standardized.
- **Floating Point Timestamps:** While `timestamp` is a float, `torchdiffeq` often works with `t` as a 0-dimensional tensor. It might be better to force `timestamp` to be a `torch.Tensor` or at least ensure compatibility early on.
- **Batch Dimension Omission:** The current state stores a single `hidden_state`. For ODE solvers used in training, batch processing is critical. `ReasoningState` should likely handle batched hidden states (e.g., shape `[Batch, Hidden]`).
- **Derivative Metadata:** For `torchdiffeq`, we will eventually need a way to pass the *derivative* of the state. It might be helpful to include a `velocity` or `delta` field in the state object if we want to store trajectory gradients.
- **Missing Integration with torch.nn:** To use this in an ODE dynamic, `ReasoningState` will likely be the "y" variable in `dy/dt = f(t, y)`. We need a clear way to flatten/unflatten this state if it includes multiple fields like `confidence` and `uncertainty`.

### Recommendations for Step 3 (ODE Reasoning Dynamics)
1. Add a `to_tensor()` and `from_tensor()` method to `ReasoningState` to facilitate passing state to/from the ODE solver.
2. Consider converting `ReasoningState` to a `NamedTuple` or a more robust `pydantic` model for easier serialization and validation.
3. Explicitly define the batch shape expectations in the documentation or type hints.
