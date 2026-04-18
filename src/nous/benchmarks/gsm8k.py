"""GSM8K benchmark — grade-school math word problems."""
from __future__ import annotations
import re
from .base import Benchmark

# Minimal built-in samples for offline testing (real GSM8K has 1.3k test items)
_BUILTIN_SAMPLES = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "answer": "18"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "answer": "540"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?", "answer": "20"},
    {"question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?", "answer": "64"},
    {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "answer": "5"},
    {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "answer": "6"},
    {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "answer": "39"},
    {"question": "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?", "answer": "33"},
]


class GSM8KBenchmark(Benchmark):
    name = "gsm8k"

    def load_samples(self) -> list[dict]:
        # Try to load from HuggingFace datasets if available
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
            samples = [{"question": row["question"], "answer": row["answer"]} for row in ds]
            return samples
        except Exception:
            return _BUILTIN_SAMPLES

    def predict(self, sample: dict) -> str:
        q = sample["question"]
        prompt = (
            "Solve this math problem step by step. "
            "At the very end write 'ANSWER: <number>' with just the numeric answer.\n\n"
            f"Problem: {q}\n\nSolution:"
        )
        return self.model.generate(prompt, max_tokens=400, temperature=0.1)

    def evaluate_sample(self, sample: dict, prediction: str) -> bool:
        expected = self._extract_number(str(sample["answer"]))
        predicted = self._extract_number(prediction)
        if expected is None or predicted is None:
            return False
        return abs(expected - predicted) < 0.01

    def _extract_number(self, text: str) -> float | None:
        patterns = [
            r"ANSWER:\s*\$?([\d,]+(?:\.\d+)?)",          # ANSWER: 30
            r"####\s*([\d,]+(?:\.\d+)?)",                  # #### 30 (GSM8K)
            r"\\boxed\{([\d,]+(?:\.\d+)?)\}",             # \boxed{30}
            r"final answer is[:\s]+\$?([\d,]+(?:\.\d+)?)", # final answer is 30
            r"answer is[:\s]+\$?([\d,]+(?:\.\d+)?)",       # answer is 30
            r"=\s*\$?([\d,]+(?:\.\d+)?)\s*$",             # = 30 at end of line
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    pass
        # Last number in text
        numbers = re.findall(r"\b(\d{1,6}(?:,\d{3})*(?:\.\d+)?)\b", text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
        return None
