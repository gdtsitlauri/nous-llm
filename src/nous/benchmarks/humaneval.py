"""HumanEval benchmark — Python code generation with execution-based critique."""
from __future__ import annotations
import ast
import re
import subprocess
import sys
import tempfile
from .base import Benchmark

_BUILTIN_SAMPLES = [
    {
        "task_id": "HumanEval/0",
        "prompt": 'def has_close_elements(numbers: list, threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
        "test": 'def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n',
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": 'def separate_paren_groups(paren_string: str) -> list:\n    """Input is a string of multiple groups of nested parentheses. Your goal is to separate those groups into separate strings and return the list of them. Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n    >>> separate_paren_groups("( ) (( )) (( )( ))")\n    ["()", "(())", "(()())"]\n    """\n',
        "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == \"(\":\n            current_depth += 1\n            current_string.append(c)\n        elif c == \")\":\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(\"\".join(current_string))\n                current_string = []\n    return result\n",
        "test": 'def check(candidate):\n    assert candidate("(()()) ((())) () ((())()())") == ["(()())", "((()))", "()", "((())()())"]\n    assert candidate("() (()) ((())) (((())))") == ["()", "(())", "((()))", "(((())))"]\n    assert candidate("(()(())((())))") == ["(()(())((())))"]\n',
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": 'def truncate_number(number: float) -> float:\n    """Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1). Return the decimal part.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
        "canonical_solution": "    return number % 1.0\n",
        "test": 'def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n',
        "entry_point": "truncate_number",
    },
]


class HumanEvalBenchmark(Benchmark):
    name = "humaneval"

    def load_samples(self) -> list[dict]:
        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
            return list(ds)
        except Exception:
            return _BUILTIN_SAMPLES

    def predict(self, sample: dict) -> str:
        prompt_text = sample["prompt"]
        entry = sample["entry_point"]
        prompt = (
            f"Complete the following Python function. "
            f"Return the complete function including the def line.\n\n"
            f"{prompt_text}"
        )
        raw = self.model.generate(prompt, max_tokens=512, temperature=0.2)
        return self._extract_function(raw, entry)

    def predict_with_critique(self, sample: dict) -> str:
        """Generate code then refine using execution-based critique."""
        from ..modules.code_critique import CodeCritique
        code = self.predict(sample)
        critique = CodeCritique(self.model, max_iter=3)
        result = critique.refine(
            prompt=sample["prompt"],
            code=code,
            test=sample["test"],
            entry_point=sample["entry_point"],
        )
        return result.history[-1]  # best (last) version

    def evaluate_sample(self, sample: dict, prediction: str) -> bool:
        if f"def {sample['entry_point']}" not in prediction:
            full_code = sample["prompt"] + prediction + "\n\n"
        else:
            full_code = prediction + "\n\n"
        full_code += sample["test"] + f"\ncheck({sample['entry_point']})\n"
        return self._run_code(full_code)

    def _extract_function(self, raw: str, entry_point: str) -> str:
        fence = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1)
        fn_match = re.search(rf"(def {re.escape(entry_point)}.*)", raw, re.DOTALL)
        if fn_match:
            return fn_match.group(1).strip()
        return raw.strip()

    def _run_code(self, code: str, timeout: int = 10) -> bool:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
