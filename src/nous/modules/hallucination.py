"""Module 6 — Hallucination Reduction: confidence calibration, fact verification, self-improving."""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel
    from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = [
    "i'm not sure", "i think", "it might be", "possibly", "probably",
    "i believe", "i'm uncertain", "i don't know", "unclear",
]

CONFIDENT_PHRASES = [
    "definitely", "certainly", "always", "never", "the fact is",
    "it is known", "obviously", "clearly", "is exactly",
]


@dataclass
class ConfidenceReport:
    raw_response: str
    confidence: float               # overall 0..1
    dimension_scores: dict[str, float]
    uncertain_claims: list[str]
    verified_claims: list[str]
    unverifiable_claims: list[str]
    recommendation: str             # "accept" | "refine" | "reject"

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.65


class HallucinationDetector:
    def __init__(self, model: "NousModel", kg: "KnowledgeGraph"):
        self.model = model
        self.kg = kg
        self._calibration_history: list[tuple[float, bool]] = []  # (predicted_conf, was_correct)

    # ------------------------------------------------------------------ #
    def analyze(self, question: str, response: str) -> ConfidenceReport:
        # 1. Linguistic signals
        linguistic_conf = self._linguistic_confidence(response)
        # 2. KG verification
        verified, unverifiable = self._kg_verify(response)
        kg_conf = len(verified) / max(1, len(verified) + len(unverifiable)) if (verified or unverifiable) else 0.5
        # 3. Self-consistency check
        consistency_conf = self._consistency_check(question, response)
        # 4. Uncertainty detection
        uncertain_claims = self._extract_uncertain_claims(response)

        overall = 0.4 * linguistic_conf + 0.3 * kg_conf + 0.3 * consistency_conf

        if overall >= 0.75:
            rec = "accept"
        elif overall >= 0.50:
            rec = "refine"
        else:
            rec = "reject"

        return ConfidenceReport(
            raw_response=response,
            confidence=overall,
            dimension_scores={
                "linguistic": linguistic_conf,
                "kg_verification": kg_conf,
                "consistency": consistency_conf,
            },
            uncertain_claims=uncertain_claims,
            verified_claims=verified,
            unverifiable_claims=unverifiable,
            recommendation=rec,
        )

    def refine_with_confidence(self, question: str, response: str, report: ConfidenceReport) -> str:
        """Rewrite response to be appropriately hedged."""
        if report.is_reliable:
            return response
        uncertain = "; ".join(report.uncertain_claims[:3]) if report.uncertain_claims else "several claims"
        prompt = f"""The following response has uncertain claims: {uncertain}
Confidence score: {report.confidence:.2f}

Original question: {question}
Original response: {response}

Rewrite the response to:
1. Clearly mark uncertain claims with "I believe" or "This may"
2. Remove overconfident statements
3. Add appropriate caveats
4. Retain all correct information

Refined response:"""
        return self.model.generate(prompt, max_tokens=512, temperature=0.4)

    def calibration_error(self) -> float:
        """Expected Calibration Error (ECE) — lower is better."""
        if len(self._calibration_history) < 5:
            return float("nan")
        buckets: dict[int, list[tuple[float, bool]]] = {}
        for conf, correct in self._calibration_history:
            bucket = int(conf * 10)
            buckets.setdefault(bucket, []).append((conf, correct))
        ece = 0.0
        n = len(self._calibration_history)
        for items in buckets.values():
            avg_conf = sum(c for c, _ in items) / len(items)
            accuracy = sum(1 for _, ok in items if ok) / len(items)
            ece += len(items) / n * abs(avg_conf - accuracy)
        return ece

    def record_outcome(self, predicted_confidence: float, was_correct: bool):
        self._calibration_history.append((predicted_confidence, was_correct))

    # ------------------------------------------------------------------ #
    def _linguistic_confidence(self, text: str) -> float:
        text_lower = text.lower()
        uncertain_count = sum(1 for p in UNCERTAINTY_PHRASES if p in text_lower)
        overconfident_count = sum(1 for p in CONFIDENT_PHRASES if p in text_lower)
        base = 0.7
        base -= uncertain_count * 0.08
        base -= overconfident_count * 0.05   # overconfidence also penalised
        return float(max(0.1, min(1.0, base)))

    def _kg_verify(self, response: str) -> tuple[list[str], list[str]]:
        """Check key claims against KG."""
        sentences = [s.strip() for s in re.split(r"[.!?]", response) if len(s.strip()) > 20][:5]
        verified, unverifiable = [], []
        for sent in sentences:
            # Extract main noun for KG lookup
            words = sent.split()[:4]
            concept = " ".join(w.lower() for w in words if len(w) > 3)[:30]
            result = self.kg.query(concept, depth=1)
            if result.get("found"):
                verified.append(sent[:80])
            else:
                unverifiable.append(sent[:80])
        return verified, unverifiable

    def _consistency_check(self, question: str, response: str) -> float:
        """Ask model if the response is internally consistent."""
        prompt = f"""Is this response internally consistent and self-contradictory?
Question: {question}
Response: {response[:400]}

Answer with ONLY a number 0.0 to 1.0 where 1.0 = perfectly consistent:"""
        raw = self.model.generate(prompt, max_tokens=10, temperature=0.1)
        match = re.search(r"([01](?:\.\d+)?|\.\d+)", raw)
        if match:
            try:
                return float(max(0.0, min(1.0, float(match.group(1)))))
            except ValueError:
                pass
        return 0.5

    def _extract_uncertain_claims(self, text: str) -> list[str]:
        sentences = re.split(r"[.!?]", text)
        uncertain = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(p in sent_lower for p in UNCERTAINTY_PHRASES):
                uncertain.append(sent.strip()[:100])
        return uncertain[:5]
