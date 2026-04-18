"""Base model loader — uses llama-cpp-python with CUDA offload."""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_INSTANCE: Optional["NousModel"] = None


class NousModel:
    """Thin wrapper around llama_cpp.Llama with prompt helpers."""

    def __init__(self, cfg=None):
        from llama_cpp import Llama
        from .config import NousConfig, DEFAULT_CONFIG

        self.cfg = cfg or DEFAULT_CONFIG
        mc = self.cfg.model
        model_path = Path(mc.model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).parent.parent.parent / model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download a GGUF model and set NOUS_MODEL_PATH or place at models/llama-3.2-3b-instruct-q4_k_m.gguf"
            )

        logger.info("Loading model %s  gpu_layers=%d", model_path, mc.n_gpu_layers)
        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=mc.n_gpu_layers,
            n_ctx=mc.n_ctx,
            n_batch=mc.n_batch,
            seed=mc.seed,
            verbose=mc.verbose,
            logits_all=False,
        )
        logger.info("Model loaded.")

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        mc = self.cfg.model
        out = self._llm(
            prompt,
            max_tokens=max_tokens or mc.max_tokens,
            temperature=temperature if temperature is not None else mc.temperature,
            top_p=mc.top_p,
            repeat_penalty=mc.repeat_penalty,
            stop=["</s>", "[INST]", "###"],
        )
        return out["choices"][0]["text"].strip()

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """messages: list of {"role": ..., "content": ...}"""
        prompt = self._format_chat(messages)
        return self.generate(prompt, max_tokens=max_tokens)

    def _format_chat(self, messages: list[dict]) -> str:
        """LLaMA-3 instruct format."""
        parts = ["<|begin_of_text|>"]
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)


def get_model(cfg=None) -> NousModel:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = NousModel(cfg)
    return _INSTANCE


def reset_model():
    global _INSTANCE
    _INSTANCE = None
