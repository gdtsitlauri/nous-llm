"""Integration test for NOUS-EVOLVE loop — mock model."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CRITIQUE_JSON = '{"accuracy": 0.8, "logic": 0.8, "completeness": 0.8, "clarity": 0.8, "conciseness": 0.8, "weaknesses": [], "suggestions": []}'
KG_JSON = '[{"subject": "test", "relation": "is_a", "object": "example", "confidence": 0.9}]'
GAP_JSON = '["concept_a", "concept_b"]'
QA_JSON = '["What is X?", "What is Y?"]'


class TestNousEvolveLoop(unittest.TestCase):
    def _make_model(self):
        model = MagicMock()
        # Return different responses based on call number
        call_responses = [
            "Initial answer to the question.",   # generate initial response
            CRITIQUE_JSON,                         # critique
            "0.8",                                 # consistency check
            KG_JSON,                               # kg extract
            GAP_JSON,                              # identify gaps
        ]
        model.generate.side_effect = lambda *a, **kw: call_responses.pop(0) if call_responses else "default response"
        return model

    def test_process_single(self):
        """Test that process_single completes without error."""
        import tempfile
        from nous.evolve import NousEvolve
        from nous.config import NousConfig
        from nous.modules.knowledge_graph import KnowledgeGraph

        model = MagicMock()
        model.generate.return_value = CRITIQUE_JSON

        cfg = NousConfig()
        nous = NousEvolve.__new__(NousEvolve)
        nous.cfg = cfg
        nous.model = model

        # Replace all modules with mocks
        from nous.modules import (
            SelfCritique, KnowledgeGraph, CuriosityEngine, MemoryStore,
            MetaLearner, HallucinationDetector, AdversarialEngine, ReasoningEngine, CodeEngine,
        )
        from nous.nlp import NLPEngine
        from nous.modules.self_critique import CritiqueResult, RefinementHistory

        mock_critique = MagicMock()
        mock_history = RefinementHistory(question="test")
        mock_cr = CritiqueResult(
            scores={"accuracy": 0.8, "logic": 0.8, "completeness": 0.8, "clarity": 0.8, "conciseness": 0.8},
            overall=0.8, weaknesses=[], suggestions=[], iteration=0
        )
        mock_history.add("Good answer.", mock_cr)
        mock_critique.evaluate_and_improve.return_value = mock_history

        from nous.modules.hallucination import ConfidenceReport
        mock_hall = MagicMock()
        mock_hall.analyze.return_value = ConfidenceReport(
            raw_response="test", confidence=0.8,
            dimension_scores={}, uncertain_claims=[], verified_claims=[],
            unverifiable_claims=[], recommendation="accept"
        )

        nous.critique = mock_critique
        nous.kg = MagicMock()
        nous.kg.extract_and_add.return_value = [("a", "b", "c")]
        nous.kg.query.return_value = {"found": False}
        nous.memory = MagicMock()
        nous.memory.retrieve.return_value = []
        nous.curiosity = MagicMock()
        nous.meta = MagicMock()
        nous.meta.classify_domain.return_value = "general"
        nous.meta.get_strategy.return_value = {"domain": "general", "learning_rate": 0.5, "recommended_iterations": 3, "temperature": 0.6, "focus_prompt": "Be accurate."}
        nous.hallucination = mock_hall
        nous.adversarial = MagicMock()
        nous.reasoning = MagicMock()
        nous.code = MagicMock()
        nous.nlp = MagicMock()

        response, meta = nous.process_single("What is 2 + 2?")
        self.assertIsInstance(response, str)
        self.assertIn("score", meta)
        self.assertIn("confidence", meta)
        self.assertIn("domain", meta)

    def test_run_session(self):
        """Test that run() processes multiple questions."""
        from nous.evolve import NousEvolve, EvolveSession
        from nous.modules.self_critique import CritiqueResult, RefinementHistory
        from nous.modules.hallucination import ConfidenceReport

        model = MagicMock()
        model.generate.return_value = "A good answer."

        nous = NousEvolve.__new__(NousEvolve)
        from nous.config import NousConfig
        nous.cfg = NousConfig()
        nous.model = model

        mock_history = RefinementHistory(question="test")
        mock_cr = CritiqueResult(
            scores={d: 0.75 for d in ["accuracy", "logic", "completeness", "clarity", "conciseness"]},
            overall=0.75, weaknesses=[], suggestions=[], iteration=0
        )
        mock_history.add("Answer.", mock_cr)

        nous.critique = MagicMock()
        nous.critique.evaluate_and_improve.return_value = mock_history
        nous.critique.get_improvement_stats.return_value = {}

        nous.kg = MagicMock()
        nous.kg.extract_and_add.return_value = []
        nous.kg.stats.return_value = {}

        nous.memory = MagicMock()
        nous.memory.retrieve.return_value = []
        nous.memory.stats.return_value = {}

        nous.curiosity = MagicMock()
        nous.curiosity.stats.return_value = {}

        nous.meta = MagicMock()
        nous.meta.classify_domain.return_value = "general"
        nous.meta.get_strategy.return_value = {"domain": "general", "learning_rate": 0.5, "recommended_iterations": 3, "temperature": 0.6, "focus_prompt": "Be accurate."}
        nous.meta.record_outcome.return_value = None
        nous.meta.learning_efficiency.return_value = {}

        nous.hallucination = MagicMock()
        mock_report = ConfidenceReport(
            raw_response="", confidence=0.8, dimension_scores={},
            uncertain_claims=[], verified_claims=[], unverifiable_claims=[],
            recommendation="accept"
        )
        nous.hallucination.analyze.return_value = mock_report

        nous.adversarial = MagicMock()
        nous.adversarial.stats.return_value = {}

        nous.reasoning = MagicMock()
        nous.reasoning.stats.return_value = {}

        nous.code = MagicMock()
        nous.code.stats.return_value = {}

        nous.nlp = MagicMock()

        questions = ["Q1?", "Q2?", "Q3?"]
        session = nous.run(questions, max_wall_minutes=5.0)

        self.assertIsInstance(session, EvolveSession)
        self.assertEqual(len(session.steps), 3)
        self.assertGreater(session.avg_score, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
