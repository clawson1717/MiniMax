# beam-draft-rl Usage Report

## Summary
The beam-draft-rl project is a well-structured PEFT-RL framework for training compact LLMs on physics reasoning. All 39 tests pass. CLI works for happy paths but has several UX issues with edge cases. The API layer has a notable bug in the Corrector class where it doesn't parse Final Answer before comparison.

## Test Suite
**Result: 39/39 tests pass** (6.92s)

Tests cover physics engine, CLI, trainer, benchmark, model wrapper, data generation, curriculum, sensor, and regulation.

## CLI Testing

### train
- `--help`: ✅ Works, shows all options
- Valid args: ✅ Works with `--model-name gpt2`
- Missing `--model-name`: ✅ Proper error message
- Empty model name (`--model-name ""`): ❌ **CRASH** - Raw HuggingFace traceback, no validation
- Negative learning rate: ❌ **SILENT ACCEPT** - No validation, just runs
- Zero learning rate: ❌ **SILENT ACCEPT** - No validation
- Invalid lr type (`--lr abc`): ✅ Proper argparse error

### evaluate
- `--help`: ✅ Works
- Missing `--model-path`: ✅ Proper error
- Valid path: ✅ Works (simulated)
- Empty path (`--model-path ""`): ❌ **SILENT ACCEPT** - Just prints "Starting evaluation of model: "

### visualize-drafts
- `--help`: ✅ Works
- No args: ✅ Works, shows demo output
- Output shows draft extraction correctly

### solve
- `--help`: ✅ Works
- Valid prompt: ✅ Works (simulated)
- Empty prompt (`solve ""`): ❌ **SILENT ACCEPT** - Prints "Solving problem: "
- Missing prompt: ✅ Proper error

### General CLI
- No subcommand: ✅ Shows help
- Invalid subcommand: ✅ Proper error with valid choices listed

## API Testing

### PhysicsEngine
- `solve_simply_supported_beam`: ✅ Works correctly
- `validate_equilibrium`: ✅ Works correctly
- **BUG**: Negative length accepted without validation, returns physically meaningless result
- Edge case: position at 0 or length works correctly
- Zero force handled correctly (returns zero reactions)

### VerifiableReward
- `calculate_reward`: ✅ Works for correct/wrong answers
- Missing "Final Answer:" pattern: ✅ Returns 0.0
- `verify_equations`: ✅ Works for valid/invalid equations
- **BUG**: `calculate_reward(None, '10kN')` raises TypeError (no None handling)

### DraftWrapper
- `wrap_prompt`: ✅ Works
- `extract_draft`: ✅ Works for valid cases
- Edge cases handled: empty string returns "", no tags returns "", unclosed tag returns ""
- `compress_cot`: ✅ Basic truncation works

### UncertaintySensor
- `calculate_entropy`: ✅ Works correctly
- `check_semantic_consistency`: ✅ Works correctly
- `detect_noise`: ✅ Works correctly
- Edge cases: Empty arrays/lists handled with safe defaults
- **QUIRK**: Entropy for [1.0, 0.0] returns -0.0 (negative zero, not a bug)

### CurriculumManager
- Stage progression: ✅ Works correctly with proper sample thresholds
- `should_advance`: ✅ Requires both min samples AND window performance
- `modify_prompt`: ✅ Different prompts per stage
- Edge case: mastery_threshold > 1 prevents advancement (correct behavior)
- Edge case: negative threshold doesn't auto-advance (accuracy still must exceed)

### Regulator
- `adjust_sampling`: ✅ Works correctly
- Clamps values to [0, 1] range: ✅
- Linear interpolation between base and min values: ✅

### Corrector
- `generate_feedback`: ❌ **MAJOR BUG**
  - The method compares `reasoning_output.strip().lower()` directly to `ground_truth.strip().lower()`
  - If reasoning_output is "Final Answer: 10kN" and ground_truth is "10kN", it says INCORRECT
  - Only exact string matches work: `generate_feedback([0.1], '10kN', '10kN')` → Correct
  - The code doesn't parse the Final Answer before comparing!

## Edge Cases

### Handled Well
- Empty arrays in sensor functions return safe defaults (0.0, False)
- Regulator clamps uncertainty scores to [0, 1]
- DraftWrapper handles missing tags gracefully

### Not Handled (Bugs)
1. None inputs cause crashes in VerifiableReward
2. Negative length accepted by PhysicsEngine
3. Empty strings accepted by CLI without validation
4. Corrector doesn't parse Final Answer before comparison

## Bugs Found

1. **CLI: Empty model name crashes with raw traceback**
   - Steps: `./venv/bin/python -m src.cli train --model-name ""`
   - Expected: Clean validation error
   - Actual: OSError from HuggingFace Hub

2. **CLI: Negative/zero learning rate silently accepted**
   - Steps: `./venv/bin/python -m src.cli train --model-name gpt2 --lr -1.0`
   - Expected: Validation error
   - Actual: "Training initiated (simulated)" with bad param

3. **PhysicsEngine: Negative length accepted**
   - Steps: `solve_simply_supported_beam('-5m', '10kN', '2m')`
   - Expected: ValueError
   - Actual: Returns nonsensical reaction values

4. **Corrector: Wrong comparison logic for Final Answer**
   - Steps: `generate_feedback([0.1], 'Final Answer: 10kN', '10kN')`
   - Expected: "Correct answer..."
   - Actual: "Incorrect answer. Ground truth: '10kN'..."
   - Root cause: Line 34 compares full reasoning_output to ground_truth without parsing

5. **VerifiableReward: None input crashes**
   - Steps: `calculate_reward(None, '10kN')`
   - Expected: Return 0.0 or clean error
   - Actual: TypeError

## Issues Found

1. **UX: Empty prompt in solve command is accepted**
   - The CLI should validate that the prompt is non-empty

2. **UX: Empty model-path in evaluate is accepted**
   - Should validate model path exists or is non-empty

3. **Documentation: No docstrings for CLI functions**
   - handle_train, handle_evaluate, etc. have no documentation

## Good Surprises

1. **Excellent test coverage**: 39 tests covering all major components
2. **Clean architecture**: Separation of concerns is well done
3. **Curriculum logic is solid**: Proper thresholds and stage progression
4. **Sensor safety**: Empty inputs handled gracefully in most places
5. **CLI help text is clear**: Good descriptions for all arguments

## Suggested GitHub Issues

### Issue 1: Add input validation for CLI numeric arguments
**Labels: bug, cli, good first issue**

The CLI accepts invalid numeric arguments without validation:
- Negative/zero learning rates are silently accepted
- Empty model names cause raw HuggingFace tracebacks

Suggested fix: Add argparse validation or post-parse checks in handle_train():
```python
if args.lr <= 0:
    parser.error("Learning rate must be positive")
if not args.model_name.strip():
    parser.error("Model name cannot be empty")
```

### Issue 2: PhysicsEngine should validate positive beam length
**Labels: bug, physics**

PhysicsEngine.solve_simply_supported_beam() accepts negative lengths without error, returning physically meaningless results.

Steps to reproduce:
```python
from src.engine import PhysicsEngine
engine = PhysicsEngine()
engine.solve_simply_supported_beam('-5m', '10kN', '2m')
# Returns {'R0': -14kN, 'RL': 4kN} instead of raising ValueError
```

Expected: Raise ValueError for negative or zero beam length.

### Issue 3: Corrector doesn't parse Final Answer before comparison
**Labels: bug, critical**

The Corrector.generate_feedback() method compares the full reasoning_output string against ground_truth instead of extracting the Final Answer first.

Steps to reproduce:
```python
from src.corrector import Corrector
c = Corrector()
c.generate_feedback([0.1], 'Final Answer: 10kN', '10kN')
# Returns: "Incorrect answer. Ground truth: '10kN'..."
# Expected: "Correct answer with high confidence..."
```

Root cause: Line 34 in corrector.py:
```python
is_correct = reasoning_output.strip().lower() == ground_truth.strip().lower()
```

Should parse Final Answer pattern first, similar to VerifiableReward.

### Issue 4: VerifiableReward crashes on None input
**Labels: bug**

VerifiableReward.calculate_reward() raises TypeError when reasoning_output is None.

Steps to reproduce:
```python
from src.reward import VerifiableReward
vr = VerifiableReward()
vr.calculate_reward(None, '10kN')
# TypeError: 'NoneType' object has no attribute 'strip'
```

Expected: Return 0.0 or raise a clean ValueError with helpful message.

### Issue 5: CLI should validate non-empty required arguments
**Labels: ux, cli**

Several CLI commands accept empty strings for required arguments:
- `solve ""` → prints "Solving problem: "
- `evaluate --model-path ""` → prints "Starting evaluation of model: "

Suggested fix: Add validation in argument handlers to reject empty strings.
