"""NOUS command-line interface."""
from __future__ import annotations
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("nous.cli")


def main():
    parser = argparse.ArgumentParser(
        prog="nous",
        description="NOUS — Neural Omnidirectional Understanding System",
    )
    sub = parser.add_subparsers(dest="command")

    # nous chat
    chat_p = sub.add_parser("chat", help="Interactive chat with NOUS")
    chat_p.add_argument("--model-path", default=None)

    # nous ask
    ask_p = sub.add_parser("ask", help="Ask NOUS a single question with NOUS-EVOLVE")
    ask_p.add_argument("question")
    ask_p.add_argument("--model-path", default=None)

    # nous benchmark
    bench_p = sub.add_parser("benchmark", help="Run benchmarks")
    bench_p.add_argument("--suite", choices=["baseline", "nous", "all"], default="all")
    bench_p.add_argument("--max-samples", type=int, default=10)

    # nous autonomous
    auto_p = sub.add_parser("autonomous", help="Run autonomous self-improvement loop")
    auto_p.add_argument("--topic", default="mathematics")
    auto_p.add_argument("--steps", type=int, default=10)
    auto_p.add_argument("--max-minutes", type=float, default=30.0)

    args = parser.parse_args()

    if args.command == "chat":
        _run_chat(args)
    elif args.command == "ask":
        _run_ask(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "autonomous":
        _run_autonomous(args)
    else:
        parser.print_help()


def _run_chat(args):
    from .model import get_model
    from .config import NousConfig
    from .nlp import NLPEngine

    print("NOUS Chat — type 'exit' to quit\n")
    cfg = NousConfig()
    if args.model_path:
        cfg.model.model_path = args.model_path
    model = get_model(cfg)
    nlp = NLPEngine(model)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        response = nlp.chat(user_input)
        print(f"NOUS: {response}\n")


def _run_ask(args):
    from .model import get_model
    from .config import NousConfig
    from .evolve import NousEvolve

    cfg = NousConfig()
    if args.model_path:
        cfg.model.model_path = args.model_path
    model = get_model(cfg)
    nous = NousEvolve(model, cfg)
    response, meta = nous.process_single(args.question)
    print(f"\nAnswer: {response}")
    print(f"\nScore: {meta['score']:.3f} | Confidence: {meta['confidence']:.3f} | Domain: {meta['domain']}")


def _run_benchmark(args):
    import subprocess
    import os
    exp_dir = str(__file__).replace("src/nous/cli.py", "experiments")
    if args.suite == "baseline":
        subprocess.run([sys.executable, f"{exp_dir}/run_baseline.py", "--max-samples", str(args.max_samples)])
    elif args.suite == "nous":
        subprocess.run([sys.executable, f"{exp_dir}/run_nous.py", "--max-samples", str(args.max_samples)])
    else:
        subprocess.run([sys.executable, f"{exp_dir}/run_all.py", "--max-samples", str(args.max_samples)])


def _run_autonomous(args):
    from .model import get_model
    from .config import NousConfig
    from .evolve import NousEvolve

    cfg = NousConfig()
    model = get_model(cfg)
    nous = NousEvolve(model, cfg)
    logger.info("Starting autonomous loop: topic=%s steps=%d max_minutes=%.1f", args.topic, args.steps, args.max_minutes)
    session = nous.run_autonomous(
        seed_topic=args.topic,
        n_steps=args.steps,
        max_wall_minutes=args.max_minutes,
    )
    print(f"\nAutonomous session complete.")
    print(f"Steps completed: {len(session.steps)}")
    print(f"Average score: {session.avg_score:.3f}")
    print(f"Wall time: {session.elapsed_minutes:.1f} min")


if __name__ == "__main__":
    main()
