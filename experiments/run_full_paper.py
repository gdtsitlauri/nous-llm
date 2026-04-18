"""Full paper experiment: all benchmarks, multi-seed, ablation, final report."""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("full_paper")

from nous.config import NousConfig
from nous.model import get_model, reset_model
from nous.evolve import NousEvolve
from nous.benchmarks import GSM8KBenchmark, HumanEvalBenchmark, TruthfulQABenchmark, MMLUBenchmark

BENCHMARKS = {
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "mmlu": MMLUBenchmark,
}

ABLATION_MODULES = ["critique", "hallucination", "memory", "kg"]

MAX_WALL_MIN = 55


def run_benchmark_baseline(model, bench_class, max_samples, label):
    bench = bench_class(model, max_samples=max_samples)
    result = bench.run(label=label)
    return result.score


def run_benchmark_nous(nous, bench_class, max_samples, label):
    from nous.benchmarks.humaneval import HumanEvalBenchmark
    bench = bench_class(nous.model, max_samples=max_samples)
    samples = bench.load_samples()[:max_samples]
    correct = 0
    for i, sample in enumerate(samples):
        try:
            # HumanEval: use execution-based code critique instead of NL NOUS loop
            if isinstance(bench, HumanEvalBenchmark):
                final_response = bench.predict_with_critique(sample)
            else:
                question = sample.get("question", sample.get("problem", ""))
                choices = sample.get("choices", [])
                if choices:
                    choices_text = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
                    question = f"{question}\n\nChoices:\n{choices_text}\n\nAnswer with letter only."
                final_response, _ = nous.process_single(question)
            if bench.evaluate_sample(sample, final_response):
                correct += 1
        except Exception as e:
            logger.warning("Sample %d error: %s", i, e)
    return correct / max(1, len(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=15)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--benchmarks", nargs="*", default=["gsm8k", "mmlu", "truthfulqa", "humaneval"])
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("results/ablation").mkdir(parents=True, exist_ok=True)

    global_start = time.time()
    all_results = {"seeds": {}, "averaged": {}, "ablation": {}}

    # ── Multi-seed runs ────────────────────────────────────────────────────
    seed_baseline = {b: [] for b in args.benchmarks}
    seed_nous = {b: [] for b in args.benchmarks}

    for seed in args.seeds:
        elapsed = (time.time() - global_start) / 60
        if elapsed >= MAX_WALL_MIN:
            logger.warning("Wall time reached. Stopping at seed %d.", seed)
            break

        logger.info("=" * 60)
        logger.info("SEED %d", seed)
        logger.info("=" * 60)

        cfg = NousConfig()
        cfg.model.seed = seed
        reset_model()
        model = get_model(cfg)

        seed_results = {"baseline": {}, "nous": {}}

        # Baseline
        for bname in args.benchmarks:
            elapsed = (time.time() - global_start) / 60
            if elapsed >= MAX_WALL_MIN:
                break
            logger.info("Baseline %s (seed=%d)", bname, seed)
            score = run_benchmark_baseline(model, BENCHMARKS[bname], args.max_samples, f"baseline_s{seed}")
            seed_results["baseline"][bname] = score
            seed_baseline[bname].append(score)
            logger.info("  → %.4f", score)

        # NOUS
        nous = NousEvolve(model, cfg)
        for bname in args.benchmarks:
            elapsed = (time.time() - global_start) / 60
            if elapsed >= MAX_WALL_MIN:
                break
            logger.info("NOUS %s (seed=%d)", bname, seed)
            score = run_benchmark_nous(nous, BENCHMARKS[bname], args.max_samples, f"nous_s{seed}")
            seed_results["nous"][bname] = score
            seed_nous[bname].append(score)
            logger.info("  → %.4f", score)

        all_results["seeds"][str(seed)] = seed_results

    # ── Average across seeds ───────────────────────────────────────────────
    for bname in args.benchmarks:
        b_scores = seed_baseline[bname]
        n_scores = seed_nous[bname]
        if b_scores and n_scores:
            avg_b = sum(b_scores) / len(b_scores)
            avg_n = sum(n_scores) / len(n_scores)
            all_results["averaged"][bname] = {
                "baseline": round(avg_b, 4),
                "nous": round(avg_n, 4),
                "delta": round(avg_n - avg_b, 4),
            }

    # ── Ablation (single seed, fast) ───────────────────────────────────────
    if not args.skip_ablation and (time.time() - global_start) / 60 < MAX_WALL_MIN - 10:
        logger.info("=" * 60)
        logger.info("ABLATION STUDY")
        logger.info("=" * 60)

        cfg = NousConfig()
        cfg.model.seed = 42
        reset_model()
        model = get_model(cfg)
        nous_full = NousEvolve(model, cfg)

        ablation_bench = "gsm8k"
        full_score = run_benchmark_nous(nous_full, BENCHMARKS[ablation_bench], 5, "nous_full")
        all_results["ablation"]["full"] = full_score

        for mod in ABLATION_MODULES:
            elapsed = (time.time() - global_start) / 60
            if elapsed >= MAX_WALL_MIN:
                break
            # Disable module by monkey-patching
            orig = getattr(nous_full, mod if mod != "kg" else "kg", None)
            if mod == "critique":
                nous_full.critique = _MockCritique()
            elif mod == "hallucination":
                nous_full.hallucination = _MockHallucination()
            elif mod == "memory":
                nous_full.memory = _MockMemory()
            elif mod == "kg":
                nous_full.kg = _MockKG()

            score = run_benchmark_nous(nous_full, BENCHMARKS[ablation_bench], 5, f"no_{mod}")
            all_results["ablation"][f"no_{mod}"] = score
            contribution = full_score - score
            logger.info("Ablation no_%s: score=%.4f contribution=%.4f", mod, score, contribution)

            # Restore
            if orig is not None:
                attr = mod if mod != "kg" else "kg"
                setattr(nous_full, attr, orig)

    # ── Save & print ───────────────────────────────────────────────────────
    all_results["wall_time_min"] = round((time.time() - global_start) / 60, 1)

    with open("results/full_paper_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (averaged across seeds)")
    print("=" * 60)
    for bname, scores in all_results["averaged"].items():
        print(f"{bname:15s}  baseline={scores['baseline']:.3f}  nous={scores['nous']:.3f}  Δ={scores['delta']:+.3f}")

    if all_results["ablation"]:
        print("\nABLATION (contribution per module):")
        full = all_results["ablation"].get("full", 0)
        for k, v in all_results["ablation"].items():
            if k != "full":
                print(f"  {k:20s}  score={v:.3f}  contribution={full-v:+.3f}")

    print(f"\nWall time: {all_results['wall_time_min']} min")
    print("Saved: results/full_paper_results.json")


# ── Mock classes for ablation ──────────────────────────────────────────────
class _MockCritique:
    def evaluate_and_improve(self, q, r, **kw):
        from nous.modules.self_critique import RefinementHistory, CritiqueResult
        h = RefinementHistory(question=q)
        cr = CritiqueResult(scores={d: 0.5 for d in ["accuracy","logic","completeness","clarity","conciseness"]},
                            overall=0.5, weaknesses=[], suggestions=[])
        h.add(r, cr)
        return h
    def get_improvement_stats(self): return {}

class _MockHallucination:
    def analyze(self, q, r):
        from nous.modules.hallucination import ConfidenceReport
        return ConfidenceReport(raw_response=r, confidence=0.7, dimension_scores={},
                                uncertain_claims=[], verified_claims=[], unverifiable_claims=[],
                                recommendation="accept")

class _MockMemory:
    def retrieve(self, *a, **kw): return []
    def push_working(self, *a): pass
    def store(self, *a, **kw): pass
    def stats(self): return {}

class _MockKG:
    def extract_and_add(self, *a, **kw): return []
    def save(self): pass
    def stats(self): return {}
    def query(self, *a, **kw): return {"found": False}
    def identify_gaps(self, *a): return []


if __name__ == "__main__":
    main()
