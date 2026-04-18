"""NOUS-EVOLVE: Evolutionary Value-driven Optimization and Learning via self-Validation Engine.

The main self-improvement loop integrating all 9 modules.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from .modules import (
    SelfCritique, KnowledgeGraph, CuriosityEngine, MemoryStore, MemoryType,
    MetaLearner, HallucinationDetector, AdversarialEngine, ReasoningEngine, CodeEngine,
)
from .nlp import NLPEngine

if TYPE_CHECKING:
    from .model import NousModel

logger = logging.getLogger(__name__)


@dataclass
class EvolveStep:
    iteration: int
    question: str
    initial_response: str
    final_response: str
    critique_score: float
    confidence: float
    kg_triples_added: int
    memory_stored: bool
    domain: str
    wall_time_s: float


@dataclass
class EvolveSession:
    session_id: str
    start_time: float = field(default_factory=time.time)
    steps: list[EvolveStep] = field(default_factory=list)
    total_questions: int = 0

    @property
    def avg_score(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.critique_score for s in self.steps) / len(self.steps)

    @property
    def improvement_over_time(self) -> list[float]:
        return [s.critique_score for s in self.steps]

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60


class NousEvolve:
    """The NOUS-EVOLVE algorithm — main entry point for autonomous improvement."""

    def __init__(self, model: "NousModel", cfg=None):
        from .config import DEFAULT_CONFIG
        self.cfg = cfg or DEFAULT_CONFIG
        self.model = model

        # Instantiate all 9 modules
        self.critique = SelfCritique(model)
        self.kg = KnowledgeGraph(model, path=self.cfg.kg_path)
        self.memory = MemoryStore(model, db_path=self.cfg.db_path)
        self.curiosity = CuriosityEngine(model, self.kg)
        self.meta = MetaLearner(model)
        self.hallucination = HallucinationDetector(model, self.kg)
        self.adversarial = AdversarialEngine(model)
        self.reasoning = ReasoningEngine(model)
        self.code = CodeEngine(model)
        self.nlp = NLPEngine(model)

        logger.info("NOUS-EVOLVE initialized with all 9 modules.")

    # ------------------------------------------------------------------ #
    def run(
        self,
        questions: list[str],
        session_id: str = "default",
        use_adversarial: bool = True,
        max_wall_minutes: float | None = None,
    ) -> EvolveSession:
        """Run the full NOUS-EVOLVE loop over a list of questions."""
        session = EvolveSession(session_id=session_id)
        max_mins = max_wall_minutes or self.cfg.experiment.max_wall_minutes

        for i, question in enumerate(questions):
            if session.elapsed_minutes >= max_mins:
                logger.warning("Wall time limit reached (%.1f min). Stopping.", max_mins)
                break

            step = self._process_one(question, iteration=i, use_adversarial=use_adversarial)
            session.steps.append(step)
            session.total_questions += 1

            logger.info(
                "Step %d/%d — domain=%s score=%.3f conf=%.3f kg+%d",
                i + 1, len(questions), step.domain, step.critique_score,
                step.confidence, step.kg_triples_added,
            )

        # Persist knowledge
        self.kg.save()
        return session

    def run_autonomous(
        self,
        seed_topic: str = "mathematics",
        n_steps: int = 10,
        max_wall_minutes: float = 30.0,
    ) -> EvolveSession:
        """Autonomous loop: curiosity engine picks topics, NOUS self-studies."""
        session = EvolveSession(session_id=f"autonomous_{seed_topic}")
        current_topic = seed_topic

        for step_i in range(n_steps):
            if session.elapsed_minutes >= max_wall_minutes:
                logger.warning("Autonomous: wall time reached.")
                break

            # 1. Curiosity picks next learning target
            target = self.curiosity.suggest_next(current_topic)
            questions = target.questions or self.curiosity.generate_questions(target.topic, n=2)

            if not questions:
                continue

            question = questions[0]
            current_topic = target.topic

            step = self._process_one(question, iteration=step_i, use_adversarial=False)
            session.steps.append(step)
            session.total_questions += 1

            # 6. Record curiosity gain
            self.curiosity.record_gain(target, step.critique_score)

            logger.info(
                "Autonomous step %d — topic=%s score=%.3f",
                step_i + 1, target.topic, step.critique_score,
            )

        self.kg.save()
        return session

    def process_single(self, question: str) -> tuple[str, dict]:
        """Convenience: run NOUS-EVOLVE on one question. Returns (answer, metadata)."""
        step = self._process_one(question, iteration=0)
        return step.final_response, {
            "score": step.critique_score,
            "confidence": step.confidence,
            "domain": step.domain,
            "kg_triples": step.kg_triples_added,
        }

    def all_stats(self) -> dict:
        return {
            "critique": self.critique.get_improvement_stats(),
            "knowledge_graph": self.kg.stats(),
            "curiosity": self.curiosity.stats(),
            "memory": self.memory.stats(),
            "meta_learning": self.meta.learning_efficiency(),
            "adversarial": self.adversarial.stats(),
            "reasoning": self.reasoning.stats(),
            "code": self.code.stats(),
        }

    # ------------------------------------------------------------------ #
    def _process_one(
        self, question: str, iteration: int = 0, use_adversarial: bool = True
    ) -> EvolveStep:
        t0 = time.time()

        # 1. Classify domain via meta-learner
        domain = self.meta.classify_domain(question)
        strategy = self.meta.get_strategy(domain)

        # 2. Retrieve relevant memories for context
        memories = self.memory.retrieve(question, top_k=3)
        mem_context = "\n".join(f"- {m.content}" for m in memories) if memories else ""

        # 3. Generate initial response
        prompt = self._build_prompt(question, strategy, mem_context)
        initial_response = self.model.generate(
            prompt,
            max_tokens=self.cfg.model.max_tokens,
            temperature=strategy["temperature"],
        )

        # 4. Self-critique and iterative refinement (Module 1)
        ec = self.cfg.evolve
        history = self.critique.evaluate_and_improve(
            question, initial_response,
            max_iter=ec.max_iterations,
            threshold=ec.quality_threshold,
        )
        best_response, best_critique = history.best or (initial_response, None)
        critique_score = best_critique.overall if best_critique else 0.5

        # 5. Hallucination check (Module 6) — analyze only, no refine for speed
        hall_report = self.hallucination.analyze(question, best_response)

        # 6. Adversarial self-play (Module 7) — disabled for speed in benchmark mode
        if use_adversarial and critique_score < 0.7 and False:
            adv_session = self.adversarial.run_session(question, n_rounds=2)
            adv_response = self.adversarial.get_final_response(adv_session)
            if adv_response and adv_session.avg_score_delta > 0:
                best_response = adv_response

        # 7. Update knowledge graph (Module 2)
        triples = self.kg.extract_and_add(best_response, source=question[:50])

        # 8. Store to memory (Module 4)
        self.memory.push_working(question)
        self.memory.store(
            content=f"Q: {question}\nA: {best_response[:300]}",
            key=f"qa_{hash(question)}",
            memory_type=MemoryType.EPISODIC,
            importance=min(0.9, critique_score),
        )

        # 9. Meta-learning update (Module 5)
        improvement = critique_score - (
            history.iterations[0][1].overall if history.iterations else critique_score
        )
        self.meta.record_outcome(domain, critique_score, improvement, len(history.iterations))

        return EvolveStep(
            iteration=iteration,
            question=question,
            initial_response=initial_response,
            final_response=best_response,
            critique_score=critique_score,
            confidence=hall_report.confidence,
            kg_triples_added=len(triples),
            memory_stored=True,
            domain=domain,
            wall_time_s=time.time() - t0,
        )

    def _build_prompt(self, question: str, strategy: dict, mem_context: str) -> str:
        parts = [strategy["focus_prompt"]]
        if mem_context:
            parts.append(f"\nRelevant background:\n{mem_context}")
        parts.append(f"\nQuestion: {question}\n\nAnswer:")
        return "\n".join(parts)
