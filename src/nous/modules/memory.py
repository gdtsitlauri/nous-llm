"""Module 4 — Memory Consolidation: working/episodic/semantic/procedural + pruning."""
from __future__ import annotations
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import NousModel

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryRecord:
    id: Optional[int]
    memory_type: MemoryType
    content: str
    key: str
    importance: float        # 0..1
    access_count: int
    created_at: float
    last_accessed: float
    metadata: dict


class MemoryStore:
    def __init__(self, model: "NousModel", db_path: str = "nous_memory.db"):
        self.model = model
        self.db_path = db_path
        self._working: list[str] = []    # fast in-memory buffer
        self._init_db()

    # ------------------------------------------------------------------ #
    # Working memory (in-RAM, tiny)

    def push_working(self, text: str, max_size: int = 10):
        self._working.append(text)
        if len(self._working) > max_size:
            self._working.pop(0)

    def get_working(self) -> list[str]:
        return list(self._working)

    def clear_working(self):
        self._working.clear()

    # ------------------------------------------------------------------ #
    # Persistent memory

    def store(
        self,
        content: str,
        key: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> int:
        now = time.time()
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO memories
                   (memory_type, content, key, importance, access_count,
                    created_at, last_accessed, metadata)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    memory_type.value, content, key, importance,
                    0, now, now, json.dumps(metadata or {}),
                ),
            )
            return cursor.lastrowid

    def retrieve(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        if memory_type:
            sql += " AND memory_type=?"
            params.append(memory_type.value)
        sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(top_k * 3)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._row_to_record(r) for r in rows]
        # Simple keyword filter
        query_words = set(query.lower().split())
        scored = []
        for r in records:
            content_words = set(r.content.lower().split())
            overlap = len(query_words & content_words) / max(1, len(query_words))
            scored.append((overlap + r.importance * 0.3, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:top_k]]

        # Update access counts
        ids = [r.id for r in results if r.id]
        if ids:
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id IN ({','.join('?'*len(ids))})",
                    [time.time()] + ids,
                )
        return results

    def consolidate(self, episodic_records: list[MemoryRecord]) -> int:
        """Merge episodic memories into semantic summary."""
        if not episodic_records:
            return 0
        combined = "\n".join(r.content for r in episodic_records[:10])
        prompt = f"""Summarize these episodic memories into a compact semantic fact (1-2 sentences):

{combined}

Semantic summary:"""
        summary = self.model.generate(prompt, max_tokens=150, temperature=0.3)
        avg_importance = sum(r.importance for r in episodic_records) / len(episodic_records)
        key = f"consolidated_{int(time.time())}"
        return self.store(summary, key, MemoryType.SEMANTIC, importance=min(0.9, avg_importance + 0.1))

    def prune(self, max_records: int = 1000) -> int:
        """Remove low-importance, rarely-accessed old memories."""
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count <= max_records:
                return 0
            to_delete = count - max_records
            conn.execute(
                """DELETE FROM memories WHERE id IN (
                   SELECT id FROM memories ORDER BY importance ASC, access_count ASC, last_accessed ASC LIMIT ?
                )""",
                (to_delete,),
            )
            return to_delete

    def stats(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------ #
    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                key TEXT UNIQUE,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL,
                last_accessed REAL,
                metadata TEXT DEFAULT '{}'
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _row_to_record(self, row) -> MemoryRecord:
        return MemoryRecord(
            id=row[0],
            memory_type=MemoryType(row[1]),
            content=row[2],
            key=row[3],
            importance=row[4],
            access_count=row[5],
            created_at=row[6],
            last_accessed=row[7],
            metadata=json.loads(row[8] or "{}"),
        )
