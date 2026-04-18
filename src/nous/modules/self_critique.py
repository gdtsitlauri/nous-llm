"""Module 1 — Self-Critique: multi-dimensional scoring and iterative refinement."""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)

DIMENSIONS = ["accuracy", "logic", "completeness", "clarity", "conciseness"]


@dataclass
class CritiqueResult:
    scores: dict[str, float]        # dimension -> 0..1
    overall: float
    weaknesses: list[str]
    suggestions: list[str]
    iteration: int = 0

    @property
    def passed(self) -> bool:
        return self.overall >= 0.80


@dataclass
class RefinementHistory:
    question: str
    iterations: list[tuple[str, CritiqueResult]] = field(default_factory=list)

    def add(self, response: str, critique: CritiqueResult):
        self.iterations.append((response, critique))

    @property
    def best(self) -> tuple[str, CritiqueResult] | None:
        if not self.iterations:
            return None
        return max(self.iterations, key=lambda x: x[1].overall)

    @property
    def improved(self) -> bool:
        if len(self.iterations) < 2:
            return False
        return self.iterations[-1][1].overall > self.iterations[0][1].overall


class SelfCritique:
    def __init__(self, model: "NousModel"):
        self.model = model
        self._history: list[RefinementHistory] = []

    # ------------------------------------------------------------------ #
    def critique(self, question: str, response: str, iteration: int = 0) -> CritiqueResult:
        prompt = self._build_critique_prompt(question, response)
        raw = self.model.generate(prompt, max_tokens=400, temperature=0.3)
        return self._parse_critique(raw, iteration)

    def refine(self, question: str, response: str, critique: CritiqueResult) -> str:
        prompt = self._build_refine_prompt(question, response, critique)
        return self.model.generate(prompt, max_tokens=512, temperature=0.6)

    def evaluate_and_improve(
        self, question: str, initial_response: str, max_iter: int = 5, threshold: float = 0.80
    ) -> RefinementHistory:
        history = RefinementHistory(question=question)
        response = initial_response

        for i in range(max_iter):
            critique = self.critique(question, response, iteration=i)
            history.add(response, critique)
            logger.debug("Iteration %d — overall=%.3f", i, critique.overall)

            if critique.passed:
                break
            if i < max_iter - 1:
                response = self.refine(question, response, critique)

        self._history.append(history)
        return history

    def get_improvement_stats(self) -> dict:
        if not self._history:
            return {}
        totals = {"improved": 0, "avg_iterations": 0.0, "avg_delta": 0.0}
        for h in self._history:
            if h.improved:
                totals["improved"] += 1
            totals["avg_iterations"] += len(h.iterations)
            if len(h.iterations) >= 2:
                totals["avg_delta"] += h.iterations[-1][1].overall - h.iterations[0][1].overall
        n = len(self._history)
        totals["avg_iterations"] /= n
        totals["avg_delta"] /= max(1, sum(1 for h in self._history if len(h.iterations) >= 2))
        totals["improved_pct"] = totals["improved"] / n
        return totals

    # ------------------------------------------------------------------ #
    def _build_critique_prompt(self, question: str, response: str) -> str:
        q = question[:150].replace('"', "'")
        a = response[:200].replace('"', "'")
        return f"""Rate this Q&A. Reply with ONLY a JSON object, no other text.

Q: {q}
A: {a}

Example format: {{"accuracy":0.8,"logic":0.7,"completeness":0.8,"clarity":0.9,"conciseness":0.7,"weaknesses":["too brief"],"suggestions":["add detail"]}}

Your JSON rating:
{{"""

    def _build_refine_prompt(self, question: str, response: str, critique: CritiqueResult) -> str:
        weak = "; ".join(critique.weaknesses[:3])
        sugg = "; ".join(critique.suggestions[:3])
        return f"""Improve the response addressing these weaknesses: {weak}
Suggestions: {sugg}

Question: {question}
Previous response: {response}

Write an improved response:"""

    def _parse_critique(self, raw: str, iteration: int) -> CritiqueResult:
        # Extract JSON block
        # The prompt pre-fills "{" so prepend it
        raw_fixed = "{" + raw if not raw.strip().startswith("{") else raw
        match = re.search(r"\{.*?\}", raw_fixed, re.DOTALL)
        scores = {d: 0.5 for d in DIMENSIONS}
        weaknesses: list[str] = []
        suggestions: list[str] = []

        if match:
            try:
                data = json.loads(match.group().split("}{")[0] + "}" if "}{" in match.group() else match.group())
                for d in DIMENSIONS:
                    if d in data:
                        scores[d] = float(max(0.0, min(1.0, data[d])))
                weaknesses = data.get("weaknesses", [])
                suggestions = data.get("suggestions", [])
            except (json.JSONDecodeError, ValueError):
                logger.warning("Could not parse critique JSON; using defaults")

        overall = sum(scores.values()) / len(scores)
        return CritiqueResult(
            scores=scores,
            overall=overall,
            weaknesses=weaknesses,
            suggestions=suggestions,
            iteration=iteration,
        )
