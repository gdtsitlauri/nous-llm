"""Module 2 — Knowledge Graph: auto-build, contradiction detection, gap identification."""
from __future__ import annotations
import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)


@dataclass
class KGNode:
    concept: str
    description: str
    confidence: float = 1.0
    source_count: int = 1


@dataclass
class KGEdge:
    relation: str
    confidence: float = 1.0


class KnowledgeGraph:
    def __init__(self, model: "NousModel", path: str = "nous_kg.pkl"):
        self.model = model
        self.path = Path(path)
        self.graph: nx.DiGraph = nx.DiGraph()
        self.contradictions: list[tuple[str, str, str]] = []  # (node_a, node_b, reason)
        self._load()

    # ------------------------------------------------------------------ #
    def extract_and_add(self, text: str, source: str = "") -> list[tuple[str, str, str]]:
        """Extract triples from text and add to graph. Returns list of (subj, rel, obj)."""
        prompt = f"""Extract factual knowledge triples from the text below.
Return ONLY a JSON array of objects with keys "subject", "relation", "object", "confidence" (0-1).
Extract up to 10 triples. Be concise.

Text: {text[:800]}

JSON array:"""
        raw = self.model.generate(prompt, max_tokens=300, temperature=0.2)
        triples = self._parse_triples(raw)

        added = []
        for subj, rel, obj, conf in triples:
            self._add_triple(subj, rel, obj, conf)
            added.append((subj, rel, obj))

        self._detect_contradictions()
        return added

    def query(self, concept: str, depth: int = 2) -> dict:
        """Return neighborhood of concept."""
        concept_lower = concept.lower()
        # Find best matching node
        node = self._find_node(concept_lower)
        if node is None:
            return {"found": False, "concept": concept}

        neighbors = nx.ego_graph(self.graph, node, radius=depth)
        facts = []
        for u, v, data in neighbors.edges(data=True):
            facts.append(f"{u} --[{data.get('relation', '?')}]--> {v}")

        return {
            "found": True,
            "concept": node,
            "facts": facts[:20],
            "degree": self.graph.degree(node),
        }

    def identify_gaps(self, topic: str) -> list[str]:
        """Return list of knowledge gaps around a topic."""
        known = list(self.graph.nodes())[:50]
        prompt = f"""Given these known concepts: {', '.join(known[:30])}

What important concepts related to "{topic}" are likely missing?
List up to 5 as a JSON array of strings."""
        raw = self.model.generate(prompt, max_tokens=200, temperature=0.5)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())[:5]
            except json.JSONDecodeError:
                pass
        return []

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump({"graph": self.graph, "contradictions": self.contradictions}, f)

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "contradictions": len(self.contradictions),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0,
        }

    # ------------------------------------------------------------------ #
    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                self.graph = data["graph"]
                self.contradictions = data.get("contradictions", [])
                logger.info("KG loaded: %d nodes, %d edges", *list(self.stats().values())[:2])
            except Exception as e:
                logger.warning("Could not load KG: %s", e)

    def _add_triple(self, subj: str, rel: str, obj: str, conf: float):
        subj, obj = subj.lower().strip(), obj.lower().strip()
        if not self.graph.has_node(subj):
            self.graph.add_node(subj, confidence=conf)
        if not self.graph.has_node(obj):
            self.graph.add_node(obj, confidence=conf)
        if self.graph.has_edge(subj, obj):
            # Update confidence
            self.graph[subj][obj]["confidence"] = max(self.graph[subj][obj]["confidence"], conf)
        else:
            self.graph.add_edge(subj, obj, relation=rel, confidence=conf)

    def _detect_contradictions(self):
        # Simple: detect bidirectional edges with conflicting relations
        new_contradictions = []
        for u, v, data in self.graph.edges(data=True):
            if self.graph.has_edge(v, u):
                rel_uv = data.get("relation", "")
                rel_vu = self.graph[v][u].get("relation", "")
                if rel_uv != rel_vu and (u, v, "bidirectional conflict") not in self.contradictions:
                    new_contradictions.append((u, v, f"conflicting: '{rel_uv}' vs '{rel_vu}'"))
        self.contradictions.extend(new_contradictions)

    def _find_node(self, concept: str) -> str | None:
        if concept in self.graph:
            return concept
        for node in self.graph.nodes():
            if concept in node or node in concept:
                return node
        return None

    def _parse_triples(self, raw: str) -> list[tuple[str, str, str, float]]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
            result = []
            for item in items:
                if isinstance(item, dict):
                    s = str(item.get("subject", "")).strip()
                    r = str(item.get("relation", "related_to")).strip()
                    o = str(item.get("object", "")).strip()
                    c = float(item.get("confidence", 0.8))
                    if s and o:
                        result.append((s, r, o, c))
            return result
        except (json.JSONDecodeError, ValueError):
            return []
