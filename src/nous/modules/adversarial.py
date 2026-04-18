"""Module 7 — Adversarial Self-Play: Generator vs Critic, blind spot discovery."""
from __future__ import annotations
import logging
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)

ATTACK_STRATEGIES = [
    "Find logical contradictions in the response.",
    "Identify factual errors or unsupported claims.",
    "Point out circular reasoning or invalid inferences.",
    "Find ambiguous or vague statements that could be misinterpreted.",
    "Identify missing important counterarguments.",
    "Find overconfident claims without sufficient evidence.",
    "Identify gaps in the reasoning chain.",
]


@dataclass
class AdversarialRound:
    question: str
    generator_response: str
    critic_attacks: list[str]
    defended_response: str
    score_delta: float    # improvement from adversarial pressure
    round_number: int


@dataclass
class AdversarialSession:
    rounds: list[AdversarialRound] = field(default_factory=list)

    @property
    def avg_score_delta(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(r.score_delta for r in self.rounds) / len(self.rounds)

    @property
    def blind_spots_found(self) -> list[str]:
        spots = []
        for r in self.rounds:
            spots.extend(r.critic_attacks)
        return spots


class AdversarialEngine:
    def __init__(self, model: "NousModel"):
        self.model = model
        self._sessions: list[AdversarialSession] = []
        self._discovered_biases: list[str] = []

    # ------------------------------------------------------------------ #
    def run_session(self, question: str, n_rounds: int = 3) -> AdversarialSession:
        session = AdversarialSession()
        response = self._generate(question)

        for round_num in range(n_rounds):
            # Critic attacks
            attacks = self._critic_attack(question, response)
            # Generator defends
            defended = self._generator_defend(question, response, attacks)
            # Measure improvement
            delta = self._score_improvement(question, response, defended)

            session.rounds.append(AdversarialRound(
                question=question,
                generator_response=response,
                critic_attacks=attacks,
                defended_response=defended,
                score_delta=delta,
                round_number=round_num,
            ))
            logger.debug("Adversarial round %d — delta=%.3f", round_num, delta)
            response = defended  # next round starts from defended version

        # Record discovered biases
        for attack in session.blind_spots_found[:3]:
            if attack not in self._discovered_biases:
                self._discovered_biases.append(attack)

        self._sessions.append(session)
        return session

    def get_final_response(self, session: AdversarialSession) -> str:
        if not session.rounds:
            return ""
        return session.rounds[-1].defended_response

    def discovered_biases(self) -> list[str]:
        return list(self._discovered_biases)

    def stats(self) -> dict:
        if not self._sessions:
            return {}
        return {
            "total_sessions": len(self._sessions),
            "avg_improvement": sum(s.avg_score_delta for s in self._sessions) / len(self._sessions),
            "total_blind_spots": len(self._discovered_biases),
        }

    # ------------------------------------------------------------------ #
    def _generate(self, question: str) -> str:
        prompt = f"Answer the following question thoroughly:\n\nQuestion: {question}\n\nAnswer:"
        return self.model.generate(prompt, max_tokens=400, temperature=0.7)

    def _critic_attack(self, question: str, response: str) -> list[str]:
        strategy = random.choice(ATTACK_STRATEGIES)
        prompt = f"""You are a ruthless critic. {strategy}

Question: {question}
Response to attack: {response}

List 2-3 specific weaknesses, errors, or blind spots. Be precise and actionable."""
        raw = self.model.generate(prompt, max_tokens=300, temperature=0.8)
        lines = [l.strip() for l in raw.split("\n") if len(l.strip()) > 15 and l.strip()[0] in "-•1234567890"]
        return [l.lstrip("-•0123456789.) ") for l in lines[:3]] if lines else [raw[:200]]

    def _generator_defend(self, question: str, response: str, attacks: list[str]) -> str:
        attack_text = "\n".join(f"- {a}" for a in attacks)
        prompt = f"""Improve the response addressing these specific criticisms:

{attack_text}

Question: {question}
Original response: {response}

Write an improved response that addresses each criticism:"""
        return self.model.generate(prompt, max_tokens=512, temperature=0.6)

    def _score_improvement(self, question: str, original: str, improved: str) -> float:
        prompt = f"""Compare these two responses to: {question}

Response A: {original[:300]}

Response B: {improved[:300]}

On a scale from -1.0 (B is worse) to +1.0 (B is better), how much did B improve over A?
Answer with ONLY a number:"""
        raw = self.model.generate(prompt, max_tokens=10, temperature=0.1)
        match = re.search(r"-?[01](?:\.\d+)?", raw)
        if match:
            try:
                return float(max(-1.0, min(1.0, float(match.group()))))
            except ValueError:
                pass
        return 0.0
