"""Module 9 — Code Critique: execution-based iterative refinement for code generation."""
from __future__ import annotations
import ast
import re
import subprocess
import sys
import tempfile
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)


@dataclass
class CodeCritiqueResult:
    passed: bool
    iterations: int
    error: str | None          # last execution error, None if passed
    history: list[str] = field(default_factory=list)  # code versions tried


class CodeCritique:
    """Execution-based critique loop for code generation.

    Unlike natural-language self-critique, this module:
    1. Runs the generated code against unit tests
    2. Feeds the concrete error message back to the LLM
    3. Asks the LLM to fix specifically that error
    4. Repeats until tests pass or max_iter reached

    This avoids the problem of NL critique corrupting syntactically valid code.
    """

    def __init__(self, model: "NousModel", max_iter: int = 3, timeout: int = 10):
        self.model = model
        self.max_iter = max_iter
        self.timeout = timeout

    def refine(self, prompt: str, code: str, test: str, entry_point: str) -> CodeCritiqueResult:
        """Iteratively refine code using execution feedback."""
        history = [code]
        current = code

        for i in range(self.max_iter):
            passed, error = self._execute(prompt, current, test, entry_point)
            if passed:
                logger.debug("CodeCritique: passed on iteration %d", i)
                return CodeCritiqueResult(passed=True, iterations=i + 1, error=None, history=history)

            if i < self.max_iter - 1:
                logger.debug("CodeCritique iter %d error: %s", i, error[:120] if error else "unknown")
                current = self._fix(prompt, current, error or "tests failed")
                history.append(current)

        passed, error = self._execute(prompt, current, test, entry_point)
        return CodeCritiqueResult(passed=passed, iterations=self.max_iter, error=error, history=history)

    def _execute(self, prompt: str, code: str, test: str, entry_point: str) -> tuple[bool, str | None]:
        if f"def {entry_point}" not in code:
            full = prompt + code + "\n\n"
        else:
            full = code + "\n\n"
        full += test + f"\ncheck({entry_point})\n"

        # Syntax check first
        try:
            ast.parse(full)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(full)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode == 0:
                return True, None
            stderr = (result.stderr or result.stdout or "").strip()
            return False, stderr[-500:] if len(stderr) > 500 else stderr
        except subprocess.TimeoutExpired:
            return False, "TimeoutError: execution exceeded time limit"
        except Exception as e:
            return False, str(e)

    def _fix(self, prompt: str, code: str, error: str) -> str:
        fix_prompt = (
            f"The following Python function has an error. Fix it.\n\n"
            f"Function:\n{code}\n\n"
            f"Error:\n{error}\n\n"
            f"Return ONLY the corrected complete function, no explanation."
        )
        raw = self.model.generate(fix_prompt, max_tokens=512, temperature=0.2)

        # Extract function from response
        fence = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1)

        # Find the def line matching the original entry point
        entry = re.search(r"def (\w+)\s*\(", prompt)
        if entry:
            ep = entry.group(1)
            fn = re.search(rf"(def {re.escape(ep)}.*)", raw, re.DOTALL)
            if fn:
                return fn.group(1).strip()

        return raw.strip()
