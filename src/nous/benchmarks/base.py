"""Base benchmark class."""
from __future__ import annotations
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    benchmark_name: str
    model_name: str
    score: float
    n_samples: int
    details: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            data = {
                "benchmark": self.benchmark_name,
                "model": self.model_name,
                "score": self.score,
                "n_samples": self.n_samples,
                "wall_time_s": self.wall_time_s,
                "metadata": self.metadata,
                "details": self.details[:20],  # don't store everything
            }
            json.dump(data, f, indent=2)


class Benchmark(ABC):
    name: str = "base"

    def __init__(self, model: "NousModel", max_samples: int = 50):
        self.model = model
        self.max_samples = max_samples

    @abstractmethod
    def load_samples(self) -> list[dict]:
        """Return list of sample dicts with at least 'question' and 'answer' keys."""
        ...

    @abstractmethod
    def evaluate_sample(self, sample: dict, prediction: str) -> bool:
        """Return True if prediction is correct."""
        ...

    def predict(self, sample: dict) -> str:
        """Default prediction — subclasses can override."""
        question = sample.get("question", sample.get("problem", ""))
        prompt = f"Answer concisely. Write 'ANSWER: <answer>' at the end.\n\nQuestion: {question}\n\nAnswer:"
        return self.model.generate(prompt, max_tokens=256, temperature=0.1)

    def run(self, label: str = "nous") -> BenchmarkResult:
        samples = self.load_samples()[:self.max_samples]
        t0 = time.time()
        correct = 0
        details = []

        for i, sample in enumerate(samples):
            try:
                pred = self.predict(sample)
                ok = self.evaluate_sample(sample, pred)
                if ok:
                    correct += 1
                details.append({"idx": i, "correct": ok, "pred": pred[:100]})
                if (i + 1) % 10 == 0:
                    logger.info("%s: %d/%d — acc=%.3f", self.name, correct, i + 1, correct / (i + 1))
            except Exception as e:
                logger.warning("Sample %d failed: %s", i, e)
                details.append({"idx": i, "correct": False, "error": str(e)})

        score = correct / max(1, len(samples))
        return BenchmarkResult(
            benchmark_name=self.name,
            model_name=label,
            score=score,
            n_samples=len(samples),
            details=details,
            wall_time_s=time.time() - t0,
        )
