"""Module 5 — Meta-Learning: learn-to-learn, per-domain adaptation, transfer."""
from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)

DOMAINS = [
    "mathematics", "science", "coding", "reasoning",
    "language", "history", "general",
]


@dataclass
class DomainStats:
    domain: str
    attempts: int = 0
    total_score: float = 0.0
    total_improvement: float = 0.0
    learning_rate: float = 0.5   # adaptive
    avg_iterations_to_threshold: float = 5.0

    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.attempts)

    @property
    def avg_improvement(self) -> float:
        return self.total_improvement / max(1, self.attempts)

    def update(self, score: float, improvement: float, n_iterations: int):
        self.attempts += 1
        self.total_score += score
        self.total_improvement += improvement
        # Adaptive learning rate: increase if improving fast
        if improvement > 0.1:
            self.learning_rate = min(0.95, self.learning_rate * 1.1)
        elif improvement < 0.02:
            self.learning_rate = max(0.05, self.learning_rate * 0.9)
        # Update avg iterations
        alpha = 0.2
        self.avg_iterations_to_threshold = (
            (1 - alpha) * self.avg_iterations_to_threshold + alpha * n_iterations
        )


class MetaLearner:
    def __init__(self, model: "NousModel"):
        self.model = model
        self.domains: dict[str, DomainStats] = {d: DomainStats(d) for d in DOMAINS}
        self._transfer_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def classify_domain(self, text: str) -> str:
        """Quick domain classification without LLM call (keyword-based)."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["equation", "math", "calculus", "algebra", "geometry", "proof", "integer", "prime"]):
            return "mathematics"
        if any(k in text_lower for k in ["code", "function", "algorithm", "program", "python", "class", "bug"]):
            return "coding"
        if any(k in text_lower for k in ["physics", "chemistry", "biology", "atom", "molecule", "cell", "force"]):
            return "science"
        if any(k in text_lower for k in ["logic", "if then", "deduce", "premise", "conclusion", "argument"]):
            return "reasoning"
        if any(k in text_lower for k in ["history", "war", "century", "empire", "revolution", "ancient"]):
            return "history"
        if any(k in text_lower for k in ["grammar", "translation", "language", "word", "syntax"]):
            return "language"
        return "general"

    def get_strategy(self, domain: str) -> dict:
        """Return recommended learning strategy for domain."""
        stats = self.domains.get(domain, DomainStats(domain))
        strategy = {
            "domain": domain,
            "learning_rate": stats.learning_rate,
            "recommended_iterations": max(2, math.ceil(stats.avg_iterations_to_threshold)),
            "temperature": self._domain_temperature(domain, stats),
            "focus_prompt": self._domain_focus(domain),
        }
        return strategy

    def record_outcome(self, domain: str, score: float, improvement: float, n_iterations: int):
        if domain not in self.domains:
            self.domains[domain] = DomainStats(domain)
        self.domains[domain].update(score, improvement, n_iterations)

    def transfer_knowledge(self, from_domain: str, to_domain: str) -> str:
        """Generate transfer prompt based on domain similarity."""
        cache_key = f"{from_domain}->{to_domain}"
        if cache_key in self._transfer_cache:
            return self._transfer_cache[cache_key]

        from_stats = self.domains.get(from_domain, DomainStats(from_domain))
        prompt = f"""What strategies that work well for {from_domain} problems
(where average score is {from_stats.avg_score:.2f}) can be transferred to {to_domain}?
Answer in 1-2 sentences."""
        transfer = self.model.generate(prompt, max_tokens=100, temperature=0.4)
        self._transfer_cache[cache_key] = transfer
        return transfer

    def learning_efficiency(self) -> dict:
        return {
            d: {
                "avg_score": s.avg_score,
                "avg_improvement": s.avg_improvement,
                "learning_rate": s.learning_rate,
                "attempts": s.attempts,
            }
            for d, s in self.domains.items()
            if s.attempts > 0
        }

    # ------------------------------------------------------------------ #
    def _domain_temperature(self, domain: str, stats: DomainStats) -> float:
        base = {"mathematics": 0.3, "coding": 0.3, "reasoning": 0.4, "science": 0.5}.get(domain, 0.6)
        # Lower temperature when doing well (exploit); raise when struggling (explore)
        if stats.avg_score > 0.7:
            return max(0.2, base - 0.1)
        elif stats.avg_score < 0.4:
            return min(0.9, base + 0.2)
        return base

    def _domain_focus(self, domain: str) -> str:
        focuses = {
            "mathematics": "Show step-by-step calculations. Verify each step.",
            "coding": "Write correct, tested code. Include edge cases.",
            "reasoning": "State premises clearly. Apply valid logical rules.",
            "science": "Ground claims in empirical evidence.",
            "language": "Use precise terminology.",
            "history": "Cite dates and sources when possible.",
            "general": "Be accurate and complete.",
        }
        return focuses.get(domain, focuses["general"])
