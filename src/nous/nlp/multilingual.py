"""NLP capabilities: multilingual (EN/GR/RU), summarization, QA, dialogue."""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en": "English", "el": "Greek", "ru": "Russian"}


@dataclass
class SummarizationResult:
    original_length: int
    summary: str
    compression_ratio: float
    self_eval_score: float


@dataclass
class QAResult:
    question: str
    answer: str
    confidence: float
    supporting_context: str


@dataclass
class DialogueTurn:
    role: str    # "user" | "nous"
    content: str
    language: str = "en"


class NLPEngine:
    def __init__(self, model: "NousModel"):
        self.model = model
        self._dialogue_history: list[DialogueTurn] = []

    # ------------------------------------------------------------------ #
    def detect_language(self, text: str) -> str:
        """Fast keyword-based language detection."""
        # Greek unicode range
        greek_chars = sum(1 for c in text if "\u0370" <= c <= "\u03ff" or "\u1f00" <= c <= "\u1fff")
        # Russian/Cyrillic unicode range
        cyrillic_chars = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
        total = max(1, len(text))
        if greek_chars / total > 0.1:
            return "el"
        if cyrillic_chars / total > 0.1:
            return "ru"
        return "en"

    def translate(self, text: str, target_lang: str = "en") -> str:
        lang_name = SUPPORTED_LANGUAGES.get(target_lang, "English")
        prompt = f"Translate the following text to {lang_name}. Return ONLY the translation.\n\nText: {text}\n\nTranslation:"
        return self.model.generate(prompt, max_tokens=300, temperature=0.3)

    def summarize(self, text: str, ratio: float = 0.3) -> SummarizationResult:
        max_words = max(30, int(len(text.split()) * ratio))
        prompt = f"""Summarize this text in at most {max_words} words. Be concise and capture key points.

Text: {text[:1500]}

Summary:"""
        summary = self.model.generate(prompt, max_tokens=min(300, max_words * 2), temperature=0.4)

        # Self-evaluate
        eval_prompt = f"""Rate this summary quality (0.0-1.0) for accuracy and completeness.
Original: {text[:500]}
Summary: {summary}
Score (number only):"""
        raw_score = self.model.generate(eval_prompt, max_tokens=10, temperature=0.1)
        match = re.search(r"([01](?:\.\d+)?|\.\d+)", raw_score)
        score = float(match.group(1)) if match else 0.7

        return SummarizationResult(
            original_length=len(text.split()),
            summary=summary,
            compression_ratio=len(summary.split()) / max(1, len(text.split())),
            self_eval_score=score,
        )

    def answer_question(self, question: str, context: str = "") -> QAResult:
        lang = self.detect_language(question)
        if lang != "en":
            question_en = self.translate(question, "en")
        else:
            question_en = question

        ctx_part = f"Context: {context[:800]}\n\n" if context else ""
        prompt = f"""{ctx_part}Answer the question accurately. Include a confidence score (0.0-1.0) at the end as "CONFIDENCE: X.X".

Question: {question_en}

Answer:"""
        raw = self.model.generate(prompt, max_tokens=400, temperature=0.5)

        conf_match = re.search(r"CONFIDENCE:\s*([01](?:\.\d+)?)", raw, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.6
        answer = re.sub(r"CONFIDENCE:.*$", "", raw, flags=re.IGNORECASE).strip()

        # Translate answer back if needed
        if lang != "en":
            answer = self.translate(answer, lang)

        return QAResult(
            question=question,
            answer=answer,
            confidence=confidence,
            supporting_context=context[:200],
        )

    def chat(self, user_message: str) -> str:
        lang = self.detect_language(user_message)
        self._dialogue_history.append(DialogueTurn(role="user", content=user_message, language=lang))

        # Build context from last 5 turns
        history_text = ""
        for turn in self._dialogue_history[-5:]:
            prefix = "User" if turn.role == "user" else "NOUS"
            history_text += f"{prefix}: {turn.content}\n"

        prompt = f"""You are NOUS, an intelligent AI assistant. Continue this conversation naturally.

{history_text}NOUS:"""
        response = self.model.generate(prompt, max_tokens=400, temperature=0.7)

        self._dialogue_history.append(DialogueTurn(role="nous", content=response, language=lang))
        return response

    def clear_dialogue(self):
        self._dialogue_history.clear()

    def dialogue_length(self) -> int:
        return len(self._dialogue_history)
