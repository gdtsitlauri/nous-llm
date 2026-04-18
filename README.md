# NOUS — Neural Omnidirectional Understanding System

**First open-source self-improving LLM framework for consumer hardware (4GB VRAM).**

NOUS runs the full autonomous self-improvement loop on an NVIDIA GTX 1650 using a
4-bit quantized LLaMA-3.2-3B model — no cloud, no API keys, no expensive GPU required.

---

## Results

Averaged over 3 random seeds (42, 43, 44), 5–10 samples per benchmark:

| Benchmark | Baseline | NOUS | Δ |
|-----------|----------|------|---|
| GSM8K | 0.567 | 0.367 | -0.200 |
| **HumanEval** | 0.200 | **0.600** | **+0.400** |
| **TruthfulQA** | 0.000 | **0.100** | **+0.100** |
| MMLU | 0.433 | 0.300 | -0.133 |
| **Average** | 0.300 | **0.342** | **+0.042** |

Key finding: **execution-based code critique** (Module 9) achieves +40% on HumanEval
by feeding concrete error messages back to the LLM instead of natural-language critique.

---

## NOUS-EVOLVE Algorithm

```
Input: question q
1.  domain  ← MetaLearner.classify(q)
2.  context ← Memory.retrieve(q, top_k=3)
3.  r₀      ← LLM(q, context)
4.  r, S    ← SelfCritique.refine(r₀, max_iter=3, threshold=0.80)
5.  C       ← Hallucination.analyze(q, r)
6.  KG.extract_and_add(r)
7.  Memory.store(q, r, importance=S)
8.  MetaLearner.record_outcome(domain, S)
9.  return r, S

For code tasks:
4b. r ← CodeCritique.refine(code, run_tests → error → fix, max_iter=3)
```

---

## 9 Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | Self-Critique | Multi-dim scoring (accuracy, logic, completeness, clarity, conciseness) |
| 2 | Knowledge Graph | Auto-build from responses, contradiction detection, gap identification |
| 3 | Curiosity Engine | UCB-based exploration of knowledge gaps |
| 4 | Memory | Working / Episodic / Semantic / Procedural (SQLite) |
| 5 | Meta-Learning | Per-domain strategy adaptation, transfer learning |
| 6 | Hallucination Reduction | Confidence calibration, KG fact verification |
| 7 | Adversarial Self-Play | Generator vs Critic at different temperatures |
| 8 | Reasoning & Logic | Chain-of-thought, math, causal reasoning |
| 9 | Code Understanding | Generation + **execution-based critique** (new) |

---

## Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GTX 1650 (4GB VRAM) |
| CUDA | 12.x |
| Model | LLaMA-3.2-3B-Instruct Q4_K_M |
| Python | 3.11+ |

---

## Quick Start

```bash
# Install
pip install -e ".[full]"

# Download model (~2GB)
bash scripts/download_model.sh

# Run full paper experiment
cd experiments
python run_full_paper.py --max-samples 10 --seeds 42 43 44 \
    --benchmarks gsm8k mmlu truthfulqa humaneval
```

---

## Project Structure

```
nous/
├── src/nous/
│   ├── config.py              # Global configuration
│   ├── model.py               # llama-cpp-python + CUDA loader
│   ├── evolve.py              # NOUS-EVOLVE main loop
│   ├── modules/
│   │   ├── self_critique.py   # Module 1
│   │   ├── knowledge_graph.py # Module 2
│   │   ├── curiosity_engine.py# Module 3
│   │   ├── memory.py          # Module 4
│   │   ├── meta_learning.py   # Module 5
│   │   ├── hallucination.py   # Module 6
│   │   ├── adversarial.py     # Module 7
│   │   ├── reasoning.py       # Module 8
│   │   ├── code_understanding.py # Module 9
│   │   └── code_critique.py   # Execution-based code critique
│   └── benchmarks/            # GSM8K, HumanEval, TruthfulQA, MMLU
├── experiments/
│   └── run_full_paper.py      # Full multi-seed experiment pipeline
├── paper/nous_paper.tex       # IEEE-format paper
├── tests/                     # 20 unit + integration tests
└── results/                   # Benchmark scores, ablation
```

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Novel Contributions

- **Consumer-hardware feasibility**: first end-to-end self-improving LLM on 4GB VRAM
- **Execution-based code critique**: +40% HumanEval vs baseline via error-driven refinement
- **Unified 9-module framework**: self-critique + KG + memory + adversarial in one system
- **Open reproducible baseline**: full code + multi-seed evaluation under MIT license

---

## Citation

```bibtex
@article{nous2025,
  title   = {NOUS: A Self-Improving LLM Framework for Consumer Hardware via NOUS-EVOLVE},
  author  = {NOUS Research Team},
  year    = {2026},
  url     = {https://github.com/nous-llm/nous}
}
```

---

## License

MIT — see [LICENSE](LICENSE)
