"""NOUS global configuration."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent  # repo root


@dataclass
class ModelConfig:
    # Model path or HuggingFace repo id for GGUF model
    model_path: str = os.environ.get(
        "NOUS_MODEL_PATH",
        "models/llama-3.2-3b-instruct-q4_k_m.gguf",
    )
    n_gpu_layers: int = int(os.environ.get("NOUS_GPU_LAYERS", "28"))  # push as many as possible
    n_ctx: int = 2048        # keep small — 4GB VRAM
    n_batch: int = 512
    max_tokens: int = 256    # generation cap per call
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    seed: int = 42
    verbose: bool = False


@dataclass
class EvolveConfig:
    max_iterations: int = 3          # per NOUS-EVOLVE loop
    quality_threshold: float = 0.80  # stop early if score >= this
    improvement_min_delta: float = 0.02


@dataclass
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    max_samples: int = 50            # keep benchmarks fast
    results_dir: Path = ROOT / "results"
    max_wall_minutes: int = 55       # hard stop before 1h


@dataclass
class NousConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    evolve: EvolveConfig = field(default_factory=EvolveConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    db_path: str = "nous_memory.db"
    kg_path: str = "nous_kg.pkl"
    log_level: str = "INFO"


DEFAULT_CONFIG = NousConfig()
