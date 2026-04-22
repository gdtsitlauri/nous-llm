# NOUS

**Neural Omnidirectional Understanding System**

**Author:** George David Tsitlauri  
**Contact:** gdtsitlauri@gmail.com  
**Website:** gdtsitlauri.dev  
**GitHub:** github.com/gdtsitlauri  
**Year:** 2026

NOUS is a consumer-hardware research framework for autonomous LLM self-improvement. It combines a quantized local language model with a nine-module loop covering self-critique, memory, knowledge-graph updates, hallucination control, adversarial self-play, and execution-based code critique.

## Evidence Status

| Item | Current status |
| --- | --- |
| Core codebase | Present |
| Benchmark harness | Present |
| Result persistence code | Present and improved |
| Archived benchmark outputs in this working copy | Not present |
| Safe public claim today | Architecture-complete, execution-ready, archival numbers must be regenerated |

Previous local development runs may have produced benchmark results, but they were not preserved in the current checked-in `results/` tree. This repo should therefore be presented as a serious implemented research system whose benchmark claims must come from persisted artifacts, not memory.

## Overview

NOUS studies whether a relatively small local LLM can improve its own outputs without cloud APIs or large multi-GPU infrastructure. The system is designed around a 4-bit quantized 3B model and a modular post-generation improvement loop.

### Core modules

| # | Module | Role |
| --- | --- | --- |
| 1 | Self-Critique | multi-dimensional response scoring and revision |
| 2 | Knowledge Graph | structured concept extraction and contradiction tracking |
| 3 | Curiosity Engine | target selection over knowledge gaps |
| 4 | Memory | working, episodic, semantic, and procedural stores |
| 5 | Meta-Learning | domain-level strategy adaptation |
| 6 | Hallucination Reduction | confidence and verification heuristics |
| 7 | Adversarial Self-Play | generator-vs-critic refinement |
| 8 | Reasoning \& Logic | structured reasoning helpers |
| 9 | Code Understanding | execution-based critique for code tasks |

## What Is Novel Here

- a full local self-improvement loop built for constrained hardware
- execution-based code critique rather than only natural-language critique
- an integrated architecture combining memory, KG updates, and self-critique
- a benchmark runner that now persists JSON and CSV outputs for future archival reuse

## Results And Artifact Hygiene

The repository now treats persisted files as the source of truth.

When `experiments/run_full_paper.py` is rerun successfully, it writes:

- `results/full_paper_results.json`
- `results/benchmarks/per_seed_scores.csv`
- `results/benchmarks/averaged_scores.csv`
- `results/ablation/ablation_summary.csv`

That means the missing-results problem is now explicitly handled as an artifact-persistence issue rather than left implicit.

What is *not* safe to do in the current working copy:

- quote old benchmark numbers that are not saved in `results/`
- present historical local runs as archival evidence

See [results/README.md](results/README.md) for the output layout.

## Repository Layout

```text
nous/
├── experiments/
│   └── run_full_paper.py
├── paper/
│   └── nous_paper.tex
├── results/
│   ├── benchmarks/
│   ├── ablation/
│   └── improvement_curves/
├── src/nous/
│   ├── evolve.py
│   ├── model.py
│   ├── config.py
│   ├── benchmarks/
│   └── modules/
└── tests/
```

## Reproduction

Install:

```bash
pip install -e ".[full]"
```

Run the benchmark harness:

```bash
python experiments/run_full_paper.py --max-samples 10 --seeds 42 43 44 \
    --benchmarks gsm8k mmlu truthfulqa humaneval
```

The benchmark runner now exports structured JSON and CSV artifacts so future README and paper updates can be evidence-backed.

## Real Vs Future Claims

| Category | Status |
| --- | --- |
| Architecture and module implementation | Present now |
| Benchmark harness | Present now |
| Archived benchmark numbers in this working copy | Missing |
| Consumer-hardware feasibility as a codebase | Supported |
| Strong external empirical claims | Require rerun and saved outputs |

## Limitations

- The current checked-in working copy does not preserve the historical benchmark outputs.
- Small local models are noisy critics of their own generations.
- Any future benchmark claims should be derived directly from saved JSON/CSV exports.

## Citation

```bibtex
@misc{tsitlauri2026nous,
  author = {George David Tsitlauri},
  title  = {NOUS: A Self-Improving LLM Framework for Consumer Hardware},
  year   = {2026},
  email  = {gdtsitlauri@gmail.com}
}
```

## License

MIT License.
