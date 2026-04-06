# L-TRAD: Logic-Driven Truth-Resilient Agentic Decoder

L-TRAD is a reasoning-heavy agentic system that generates and executes complex tasks with built-in causal verification. It combines "Draft-Thinking" for efficient exploration with "DenoiseFlow" for high-accuracy truth decoding.

## 🚀 The Denoised Draft Pipeline

1. **Explore:** Generate multiple reasoning paths using **Draft-Thinking** (Fast/Cheap).
2. **Sense:** Use **DenoiseFlow** to score drafts for semantic ambiguity and causal drift.
3. **Decode:** Select and "heal" the most promising path via the **TruthDecoder**.
4. **Verify:** Final state validation through **LOGIGEN** logic constraints.

## 📁 Structure

- `src/models.py`: Causal task and draft schemas.
- `src/orchestrator.py`: Triple-agent Architect for task decomposition.
- `src/explorer.py`: Explorer agent utilizing Draft-Thinking.
- `src/sensing.py`: Ambiguity and drift sensors.
- `src/decoder.py`: Truth-resilient path selection.
- `src/healing.py`: Multi-agent correction loop for noisy nodes.

## 🛠 Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the L-TRAD agent:
```bash
python -m src.agent --query "Your complex logical query here"
```

Launch the live reasoning terminal:
```bash
python -m src.display
```

---
Part of the **ClawWork** series. 🦞
