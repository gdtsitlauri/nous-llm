"""Unit tests for NOUS modules — mock-based, no model required."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_mock_model(responses: list[str] | None = None):
    model = MagicMock()
    if responses:
        model.generate.side_effect = responses
    else:
        model.generate.return_value = '{"accuracy": 0.8, "logic": 0.7, "completeness": 0.75, "clarity": 0.8, "conciseness": 0.7, "weaknesses": ["lacks detail"], "suggestions": ["add examples"]}'
    return model


class TestSelfCritique(unittest.TestCase):
    def test_critique_parses_json(self):
        from nous.modules.self_critique import SelfCritique
        model = make_mock_model()
        sc = SelfCritique(model)
        result = sc.critique("What is 2+2?", "The answer is 4.")
        self.assertIn("accuracy", result.scores)
        self.assertGreater(result.overall, 0)
        self.assertLessEqual(result.overall, 1)

    def test_critique_handles_bad_json(self):
        from nous.modules.self_critique import SelfCritique
        model = MagicMock()
        model.generate.return_value = "not valid json at all"
        sc = SelfCritique(model)
        result = sc.critique("test", "test response")
        self.assertEqual(result.overall, 0.5)

    def test_evaluate_and_improve(self):
        from nous.modules.self_critique import SelfCritique
        responses = [
            '{"accuracy": 0.9, "logic": 0.9, "completeness": 0.9, "clarity": 0.9, "conciseness": 0.9, "weaknesses": [], "suggestions": []}',
        ]
        model = make_mock_model(responses)
        sc = SelfCritique(model)
        history = sc.evaluate_and_improve("Q?", "Good answer.", max_iter=3)
        self.assertGreater(len(history.iterations), 0)
        self.assertIsNotNone(history.best)


class TestKnowledgeGraph(unittest.TestCase):
    def test_add_and_query(self):
        from nous.modules.knowledge_graph import KnowledgeGraph
        model = MagicMock()
        model.generate.return_value = '[{"subject": "python", "relation": "is_a", "object": "language", "confidence": 0.9}]'
        kg = KnowledgeGraph(model, path=":memory:_test.pkl")
        triples = kg.extract_and_add("Python is a programming language.")
        self.assertGreater(len(triples), 0)
        result = kg.query("python")
        self.assertTrue(result["found"])

    def test_stats(self):
        from nous.modules.knowledge_graph import KnowledgeGraph
        model = MagicMock()
        model.generate.return_value = '[]'
        kg = KnowledgeGraph(model, path=":memory:_test2.pkl")
        stats = kg.stats()
        self.assertIn("nodes", stats)
        self.assertIn("edges", stats)


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpfile = tempfile.mktemp(suffix=".db")

    def tearDown(self):
        import os
        try:
            os.unlink(self.tmpfile)
        except Exception:
            pass

    def test_store_and_retrieve(self):
        from nous.modules.memory import MemoryStore, MemoryType
        model = MagicMock()
        store = MemoryStore(model, db_path=self.tmpfile)
        store.store("Python is awesome", key="py1", memory_type=MemoryType.SEMANTIC, importance=0.8)
        results = store.retrieve("Python")
        self.assertGreater(len(results), 0)
        self.assertIn("Python", results[0].content)

    def test_working_memory(self):
        from nous.modules.memory import MemoryStore
        model = MagicMock()
        store = MemoryStore(model, db_path=self.tmpfile)
        store.push_working("item1")
        store.push_working("item2")
        working = store.get_working()
        self.assertIn("item1", working)
        self.assertIn("item2", working)

    def test_prune(self):
        from nous.modules.memory import MemoryStore, MemoryType
        model = MagicMock()
        store = MemoryStore(model, db_path=self.tmpfile)
        for i in range(5):
            store.store(f"fact {i}", key=f"key{i}", importance=0.3)
        deleted = store.prune(max_records=3)
        self.assertGreaterEqual(deleted, 0)


class TestMetaLearner(unittest.TestCase):
    def test_classify_domain(self):
        from nous.modules.meta_learning import MetaLearner
        model = MagicMock()
        ml = MetaLearner(model)
        self.assertEqual(ml.classify_domain("solve this equation x+5=10"), "mathematics")
        self.assertEqual(ml.classify_domain("write a python function"), "coding")
        self.assertEqual(ml.classify_domain("what is history of rome"), "history")

    def test_record_and_strategy(self):
        from nous.modules.meta_learning import MetaLearner
        model = MagicMock()
        ml = MetaLearner(model)
        ml.record_outcome("mathematics", 0.8, 0.2, 3)
        strategy = ml.get_strategy("mathematics")
        self.assertIn("learning_rate", strategy)
        self.assertIn("temperature", strategy)


class TestCuriosityEngine(unittest.TestCase):
    def test_suggest_next(self):
        from nous.modules.curiosity_engine import CuriosityEngine
        from nous.modules.knowledge_graph import KnowledgeGraph
        model = MagicMock()
        model.generate.return_value = '["What is calculus?", "How does integration work?"]'
        kg = KnowledgeGraph(model, path=":mem:_test3.pkl")
        ce = CuriosityEngine(model, kg)
        target = ce.suggest_next("mathematics")
        self.assertIsNotNone(target)
        self.assertIsInstance(target.topic, str)


class TestHallucinationDetector(unittest.TestCase):
    def test_analyze(self):
        from nous.modules.hallucination import HallucinationDetector
        from nous.modules.knowledge_graph import KnowledgeGraph
        model = MagicMock()
        model.generate.return_value = "0.8"
        kg = KnowledgeGraph(model, path=":mem:_test4.pkl")
        hd = HallucinationDetector(model, kg)
        report = hd.analyze("What is 2+2?", "The answer is 4.")
        self.assertGreater(report.confidence, 0)
        self.assertIn(report.recommendation, ["accept", "refine", "reject"])


class TestReasoningEngine(unittest.TestCase):
    def test_classify(self):
        from nous.modules.reasoning import ReasoningEngine
        model = MagicMock()
        model.generate.return_value = "Step 1: setup\nStep 2: compute 5*3=15\nANSWER: 15"
        re_engine = ReasoningEngine(model)
        self.assertEqual(re_engine._classify("solve x + 5 = 10"), "math")
        self.assertEqual(re_engine._classify("if all A are B and all B are C then"), "logic")

    def test_solve(self):
        from nous.modules.reasoning import ReasoningEngine
        model = MagicMock()
        model.generate.return_value = "Step 1: 5 * 3 = 15\nANSWER: 15\nYES"
        re_engine = ReasoningEngine(model)
        trace = re_engine.solve("What is 5 times 3?", problem_type="math")
        self.assertIsNotNone(trace.final_answer)


class TestCodeEngine(unittest.TestCase):
    def test_generate_python(self):
        from nous.modules.code_understanding import CodeEngine
        model = MagicMock()
        model.generate.return_value = "```python\ndef add(a, b):\n    return a + b\n```"
        ce = CodeEngine(model)
        result = ce.generate("Write a function that adds two numbers", language="python")
        self.assertTrue(result.syntax_valid)
        self.assertIn("def", result.code)


class TestNLPEngine(unittest.TestCase):
    def test_detect_language(self):
        from nous.nlp.multilingual import NLPEngine
        model = MagicMock()
        nlp = NLPEngine(model)
        self.assertEqual(nlp.detect_language("Hello world"), "en")
        self.assertEqual(nlp.detect_language("Γεια σου κόσμε"), "el")
        self.assertEqual(nlp.detect_language("Привет мир"), "ru")

    def test_chat(self):
        from nous.nlp.multilingual import NLPEngine
        model = MagicMock()
        model.generate.return_value = "Hello! How can I help you?"
        nlp = NLPEngine(model)
        response = nlp.chat("Hello NOUS!")
        self.assertIsInstance(response, str)
        self.assertEqual(nlp.dialogue_length(), 2)


class TestAdversarialEngine(unittest.TestCase):
    def test_session(self):
        from nous.modules.adversarial import AdversarialEngine
        model = MagicMock()
        model.generate.side_effect = [
            "Initial response to the question.",
            "- Lacks specific examples\n- Too vague",
            "Improved response with examples.",
            "0.3",
            "- Still could be better",
            "Even better response.",
            "0.2",
        ]
        ae = AdversarialEngine(model)
        session = ae.run_session("What is machine learning?", n_rounds=2)
        self.assertEqual(len(session.rounds), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
