"""Module 9 — Code Understanding: self-improving code gen, execution eval, HumanEval/MBPP."""
from __future__ import annotations
import ast
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"python", "javascript", "cpp", "cuda"}


@dataclass
class CodeResult:
    code: str
    language: str
    passed_tests: int
    total_tests: int
    execution_output: str
    syntax_valid: bool
    self_improved: bool = False
    iterations: int = 1

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / max(1, self.total_tests)

    @property
    def is_correct(self) -> bool:
        return self.pass_rate == 1.0 and self.syntax_valid


class CodeEngine:
    def __init__(self, model: "NousModel"):
        self.model = model
        self._results: list[CodeResult] = []

    # ------------------------------------------------------------------ #
    def generate(
        self,
        problem: str,
        language: str = "python",
        test_cases: list[dict] | None = None,
        max_iterations: int = 3,
    ) -> CodeResult:
        code = self._generate_code(problem, language)
        result = self._evaluate(code, language, test_cases or [])

        for i in range(1, max_iterations):
            if result.is_correct:
                break
            # Self-improve
            code = self._improve_code(problem, code, language, result)
            new_result = self._evaluate(code, language, test_cases or [])
            new_result.self_improved = True
            new_result.iterations = i + 1
            if new_result.pass_rate >= result.pass_rate:
                result = new_result

        self._results.append(result)
        return result

    def explain(self, code: str, language: str = "python") -> str:
        prompt = f"""Explain this {language} code clearly and concisely:

```{language}
{code}
```

Explanation:"""
        return self.model.generate(prompt, max_tokens=300, temperature=0.4)

    def review(self, code: str, language: str = "python") -> dict:
        prompt = f"""Review this {language} code for bugs, inefficiencies, and improvements.
Return JSON with keys: "bugs" (list), "improvements" (list), "quality_score" (0-1).

```{language}
{code}
```

JSON review:"""
        raw = self.model.generate(prompt, max_tokens=300, temperature=0.3)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                import json
                return json.loads(match.group())
            except Exception:
                pass
        return {"bugs": [], "improvements": [], "quality_score": 0.5}

    def stats(self) -> dict:
        if not self._results:
            return {}
        correct = sum(1 for r in self._results if r.is_correct)
        improved = sum(1 for r in self._results if r.self_improved)
        return {
            "total": len(self._results),
            "pass_at_1": correct / len(self._results),
            "self_improved": improved,
            "avg_iterations": sum(r.iterations for r in self._results) / len(self._results),
        }

    # ------------------------------------------------------------------ #
    def _generate_code(self, problem: str, language: str) -> str:
        lang_hints = {
            "python": "Write clean Python 3 code. Include a main function.",
            "javascript": "Write modern JavaScript (ES2020+). No external imports.",
            "cpp": "Write standard C++17 code.",
            "cuda": "Write CUDA C++ code with proper memory management.",
        }
        hint = lang_hints.get(language, "Write clean code.")
        prompt = f"""{hint}
Solve the following problem. Return ONLY the code, no explanations.

Problem: {problem}

```{language}"""
        raw = self.model.generate(prompt, max_tokens=512, temperature=0.4)
        return self._extract_code(raw, language)

    def _improve_code(self, problem: str, code: str, language: str, result: CodeResult) -> str:
        prompt = f"""The code has issues (pass rate: {result.pass_rate:.1%}):
{result.execution_output[:300]}

Fix the code to solve: {problem}

Original code:
```{language}
{code}
```

Fixed code (return ONLY code):
```{language}"""
        raw = self.model.generate(prompt, max_tokens=512, temperature=0.5)
        return self._extract_code(raw, language)

    def _evaluate(self, code: str, language: str, test_cases: list[dict]) -> CodeResult:
        syntax_valid = self._check_syntax(code, language)
        if not syntax_valid:
            return CodeResult(
                code=code, language=language,
                passed_tests=0, total_tests=max(1, len(test_cases)),
                execution_output="Syntax error", syntax_valid=False,
            )

        if language != "python" or not test_cases:
            # For non-Python or no tests: run code and check output
            output = self._run_python(code, timeout=10)
            return CodeResult(
                code=code, language=language,
                passed_tests=1 if "error" not in output.lower() else 0,
                total_tests=1,
                execution_output=output,
                syntax_valid=True,
            )

        passed = 0
        outputs = []
        for tc in test_cases[:5]:  # limit tests
            inp = tc.get("input", "")
            expected = str(tc.get("expected", ""))
            output = self._run_python_with_input(code, inp, timeout=10)
            if expected in output:
                passed += 1
            outputs.append(output[:100])

        return CodeResult(
            code=code, language=language,
            passed_tests=passed, total_tests=len(test_cases[:5]),
            execution_output="\n".join(outputs),
            syntax_valid=True,
        )

    def _check_syntax(self, code: str, language: str) -> bool:
        if language == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False
        return True  # Can't easily check other languages without compiler

    def _run_python(self, code: str, timeout: int = 10) -> str:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True, timeout=timeout
            )
            return (result.stdout + result.stderr)[:500]
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except Exception as e:
            return f"ERROR: {e}"

    def _run_python_with_input(self, code: str, inp: str, timeout: int = 10) -> str:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                input=inp, capture_output=True, text=True, timeout=timeout
            )
            return (result.stdout + result.stderr)[:300]
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except Exception as e:
            return f"ERROR: {e}"

    def _extract_code(self, raw: str, language: str) -> str:
        # Try to extract from code fence
        pattern = rf"```(?:{language})?\n?(.*?)```"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Return raw if no fence found
        return raw.strip()
