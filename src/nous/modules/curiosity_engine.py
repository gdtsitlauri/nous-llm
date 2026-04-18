"""Module 3 — Curiosity Engine: autonomous learning direction, exploration vs exploitation."""
from __future__ import annotations
import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel
    from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class LearningTarget:
    topic: str
    priority: float          # 0..1
    source: str              # "gap" | "exploit" | "explore"
    questions: list[str] = field(default_factory=list)
    attempts: int = 0
    gain: float = 0.0        # knowledge gain after studying

    def ucb_score(self, total_steps: int) -> float:
        """Upper Confidence Bound for exploration-exploitation."""
        if self.attempts == 0:
            return float("inf")
        exploit = self.gain / self.attempts
        explore = math.sqrt(2 * math.log(max(1, total_steps)) / self.attempts)
        return exploit + explore


class CuriosityEngine:
    def __init__(self, model: "NousModel", kg: "KnowledgeGraph"):
        self.model = model
        self.kg = kg
        self._targets: list[LearningTarget] = []
        self._total_steps = 0

    # ------------------------------------------------------------------ #
    def suggest_next(self, current_topic: str | None = None) -> LearningTarget:
        """Pick next learning target via UCB exploration-exploitation."""
        self._refresh_targets(current_topic)
        if not self._targets:
            return self._create_explore_target()

        # UCB selection
        best = max(self._targets, key=lambda t: t.ucb_score(self._total_steps))
        self._total_steps += 1
        best.attempts += 1
        return best

    def generate_questions(self, topic: str, n: int = 5) -> list[str]:
        prompt = f"""Generate {n} insightful self-study questions about "{topic}".
Focus on questions that reveal gaps or deepen understanding.
Return as JSON array of strings."""
        raw = self.model.generate(prompt, max_tokens=300, temperature=0.8)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                qs = json.loads(match.group())
                return [str(q) for q in qs[:n]]
            except json.JSONDecodeError:
                pass
        # Fallback: parse lines
        lines = [l.strip("- 0123456789.)").strip() for l in raw.split("\n") if len(l.strip()) > 10]
        return lines[:n]

    def record_gain(self, target: LearningTarget, gain: float):
        target.gain += gain
        logger.debug("Curiosity: topic=%s gain=%.3f total_gain=%.3f", target.topic, gain, target.gain)

    def exploration_rate(self) -> float:
        """Fraction of targets from exploration (vs exploitation)."""
        if not self._targets:
            return 1.0
        n_explore = sum(1 for t in self._targets if t.source == "explore")
        return n_explore / len(self._targets)

    def stats(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "n_targets": len(self._targets),
            "exploration_rate": self.exploration_rate(),
            "avg_gain": sum(t.gain for t in self._targets) / max(1, len(self._targets)),
        }

    # ------------------------------------------------------------------ #
    def _refresh_targets(self, current_topic: str | None):
        # Remove exhausted targets
        self._targets = [t for t in self._targets if t.attempts < 10]

        # Add gap-based targets from KG
        if current_topic:
            gaps = self.kg.identify_gaps(current_topic)
            for gap in gaps:
                if not any(t.topic == gap for t in self._targets):
                    qs = self.generate_questions(gap, n=3)
                    self._targets.append(LearningTarget(
                        topic=gap, priority=0.7, source="gap", questions=qs
                    ))

        # Add random exploration target occasionally
        if random.random() < 0.3:
            self._targets.append(self._create_explore_target())

    def _create_explore_target(self) -> LearningTarget:
        domains = [
            "mathematics", "physics", "computer science", "biology",
            "history", "philosophy", "linguistics", "chemistry",
        ]
        topic = random.choice(domains)
        qs = self.generate_questions(topic, n=3)
        return LearningTarget(topic=topic, priority=0.4, source="explore", questions=qs)
