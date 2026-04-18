"""MMLU benchmark — massive multitask language understanding."""
from __future__ import annotations
import re
from .base import Benchmark

_BUILTIN_SAMPLES = [
    {"question": "What is the derivative of sin(x)?", "choices": ["cos(x)", "-cos(x)", "tan(x)", "-sin(x)"], "answer": 0},
    {"question": "Which of the following is NOT a prime number?", "choices": ["2", "7", "9", "11"], "answer": 2},
    {"question": "The speed of light in vacuum is approximately:", "choices": ["3×10^6 m/s", "3×10^8 m/s", "3×10^10 m/s", "3×10^4 m/s"], "answer": 1},
    {"question": "Which element has atomic number 6?", "choices": ["Nitrogen", "Oxygen", "Carbon", "Boron"], "answer": 2},
    {"question": "What is the powerhouse of the cell?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"], "answer": 2},
    {"question": "Which of the following sorting algorithms has O(n log n) average time complexity?", "choices": ["Bubble sort", "Selection sort", "Insertion sort", "Merge sort"], "answer": 3},
    {"question": "In Python, which data structure uses key-value pairs?", "choices": ["List", "Tuple", "Set", "Dictionary"], "answer": 3},
    {"question": "What does CPU stand for?", "choices": ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Core Processing Unit"], "answer": 0},
    {"question": "Which philosopher is associated with the 'categorical imperative'?", "choices": ["Aristotle", "Kant", "Nietzsche", "Plato"], "answer": 1},
    {"question": "What is the SI unit of electric current?", "choices": ["Volt", "Watt", "Ampere", "Ohm"], "answer": 2},
]


class MMLUBenchmark(Benchmark):
    name = "mmlu"

    def load_samples(self) -> list[dict]:
        try:
            from datasets import load_dataset
            # Use a single subject for speed
            ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
            samples = []
            for row in ds:
                samples.append({
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],  # int 0-3
                })
            return samples
        except Exception:
            return _BUILTIN_SAMPLES

    def predict(self, sample: dict) -> str:
        choices = sample["choices"]
        choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt = (
            f"Choose the correct answer. Reply with ONLY the letter (A, B, C, or D).\n\n"
            f"Question: {sample['question']}\n\n{choices_text}\n\nAnswer:"
        )
        return self.model.generate(prompt, max_tokens=5, temperature=0.1)

    def evaluate_sample(self, sample: dict, prediction: str) -> bool:
        correct_letter = chr(65 + sample["answer"])
        pred_upper = prediction.upper()
        # Check start, or "answer is X", or standalone letter
        import re
        if pred_upper.strip().startswith(correct_letter):
            return True
        # Search for "answer: X" or "answer is X" or "(X)" patterns
        patterns = [
            rf"ANSWER[:\s]+{correct_letter}\b",
            rf"THE ANSWER IS {correct_letter}\b",
            rf"\b{correct_letter}\)",
            rf"^{correct_letter}\b",
        ]
        for p in patterns:
            if re.search(p, pred_upper):
                return True
        # Check if the correct choice text appears prominently
        choices = sample.get("choices", [])
        if sample["answer"] < len(choices):
            correct_text = choices[sample["answer"]].lower()
            if len(correct_text) > 5 and correct_text[:20] in prediction.lower():
                return True
        return False
