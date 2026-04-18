"""Module 8 — Reasoning & Logic: self-improving CoT, math, causal reasoning, GSM8K/MATH/LogiQA."""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    step_number: int
    description: str
    operation: str       # "setup" | "compute" | "verify" | "conclude"
    result: str
    confidence: float


@dataclass
class ReasoningTrace:
    problem: str
    steps: list[ReasoningStep]
    final_answer: str
    verified: bool
    error_corrected: bool = False

    @property
    def chain_of_thought(self) -> str:
        lines = []
        for s in self.steps:
            lines.append(f"Step {s.step_number} ({s.operation}): {s.description} → {s.result}")
        return "\n".join(lines)


class ReasoningEngine:
    def __init__(self, model: "NousModel"):
        self.model = model
        self._traces: list[ReasoningTrace] = []
        self._error_patterns: list[str] = []   # track systematic errors

    # ------------------------------------------------------------------ #
    def solve(self, problem: str, problem_type: str = "auto") -> ReasoningTrace:
        if problem_type == "auto":
            problem_type = self._classify(problem)

        if problem_type == "math":
            trace = self._solve_math(problem)
        elif problem_type == "logic":
            trace = self._solve_logic(problem)
        elif problem_type == "causal":
            trace = self._solve_causal(problem)
        else:
            trace = self._solve_general(problem)

        self._traces.append(trace)
        return trace

    def verify_answer(self, problem: str, answer: str) -> bool:
        prompt = f"""Does this answer correctly solve the problem?
Problem: {problem}
Answer: {answer}

Reply with only YES or NO:"""
        raw = self.model.generate(prompt, max_tokens=5, temperature=0.1)
        return "yes" in raw.lower()

    def stats(self) -> dict:
        if not self._traces:
            return {}
        verified = sum(1 for t in self._traces if t.verified)
        corrected = sum(1 for t in self._traces if t.error_corrected)
        return {
            "total_problems": len(self._traces),
            "verified_correct": verified,
            "accuracy": verified / len(self._traces),
            "self_corrections": corrected,
            "error_patterns": self._error_patterns[:5],
        }

    # ------------------------------------------------------------------ #
    def _solve_math(self, problem: str) -> ReasoningTrace:
        prompt = f"""Solve this math problem step by step.
Show each calculation explicitly. At the end write "ANSWER: <number>".

Problem: {problem}

Solution:"""
        raw = self.model.generate(prompt, max_tokens=512, temperature=0.3)
        steps = self._parse_steps(raw)
        answer = self._extract_answer(raw)
        verified = self.verify_answer(problem, answer)
        trace = ReasoningTrace(problem=problem, steps=steps, final_answer=answer, verified=verified)

        if not verified:
            # Self-correct
            corrected_trace = self._self_correct(problem, trace, "math")
            if corrected_trace.verified:
                return corrected_trace
        return trace

    def _solve_logic(self, problem: str) -> ReasoningTrace:
        prompt = f"""Solve this logic problem using explicit reasoning.
Identify premises, apply logical rules, and derive conclusion.
Write "ANSWER: <conclusion>" at the end.

Problem: {problem}

Reasoning:"""
        raw = self.model.generate(prompt, max_tokens=400, temperature=0.3)
        steps = self._parse_steps(raw)
        answer = self._extract_answer(raw)
        verified = self.verify_answer(problem, answer)
        return ReasoningTrace(problem=problem, steps=steps, final_answer=answer, verified=verified)

    def _solve_causal(self, problem: str) -> ReasoningTrace:
        prompt = f"""Analyze this causal question.
Identify: (1) cause, (2) mechanism, (3) effect, (4) confounders.
Write "ANSWER: <causal conclusion>" at the end.

Problem: {problem}

Causal analysis:"""
        raw = self.model.generate(prompt, max_tokens=400, temperature=0.4)
        steps = self._parse_steps(raw)
        answer = self._extract_answer(raw)
        verified = self.verify_answer(problem, answer)
        return ReasoningTrace(problem=problem, steps=steps, final_answer=answer, verified=verified)

    def _solve_general(self, problem: str) -> ReasoningTrace:
        prompt = f"""Think step by step to answer this question.
Write "ANSWER: <answer>" at the end.

Question: {problem}

Step-by-step:"""
        raw = self.model.generate(prompt, max_tokens=400, temperature=0.5)
        steps = self._parse_steps(raw)
        answer = self._extract_answer(raw)
        verified = self.verify_answer(problem, answer)
        return ReasoningTrace(problem=problem, steps=steps, final_answer=answer, verified=verified)

    def _self_correct(self, problem: str, trace: ReasoningTrace, ptype: str) -> ReasoningTrace:
        prompt = f"""Your previous solution was incorrect:
{trace.chain_of_thought}
Answer given: {trace.final_answer}

Find the error and solve again. Write "ANSWER: <corrected answer>" at the end.

Problem: {problem}

Corrected solution:"""
        raw = self.model.generate(prompt, max_tokens=512, temperature=0.4)
        steps = self._parse_steps(raw)
        answer = self._extract_answer(raw)
        verified = self.verify_answer(problem, answer)
        if not verified:
            # Track error pattern
            self._error_patterns.append(f"{ptype}: {problem[:50]}")
        return ReasoningTrace(
            problem=problem, steps=steps, final_answer=answer,
            verified=verified, error_corrected=True
        )

    def _classify(self, problem: str) -> str:
        p = problem.lower()
        if any(k in p for k in ["=", "solve", "calculate", "how many", "how much", "$", "percent", "sum", "product"]):
            return "math"
        if any(k in p for k in ["if", "then", "all", "some", "none", "valid", "argument", "premise"]):
            return "logic"
        if any(k in p for k in ["cause", "effect", "why", "leads to", "result of", "because"]):
            return "causal"
        return "general"

    def _parse_steps(self, raw: str) -> list[ReasoningStep]:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        steps = []
        for i, line in enumerate(lines[:10]):
            if line.lower().startswith("answer:"):
                break
            op = "compute" if any(c in line for c in "=+-*/×÷") else "setup"
            if i == 0:
                op = "setup"
            steps.append(ReasoningStep(
                step_number=i + 1,
                description=line[:100],
                operation=op,
                result=line[:50],
                confidence=0.8,
            ))
        return steps

    def _extract_answer(self, raw: str) -> str:
        match = re.search(r"ANSWER:\s*(.+)", raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Try last non-empty line
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[-1] if lines else raw[:50]
