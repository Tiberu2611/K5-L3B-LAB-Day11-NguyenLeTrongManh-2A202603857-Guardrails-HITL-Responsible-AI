"""
Assignment 11 — Audit Log (implemented).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_TEXT_CHARS = 1000  # cắt bớt để log không phình to


def _scrub(text: str) -> str:
    """Che PII/secret trước khi ghi log — audit log cũng là nơi có thể bị lộ dữ liệu."""
    from guardrails.output_guardrails import content_filter

    cleaned = content_filter(text or "")["redacted"]
    return cleaned[:MAX_TEXT_CHARS]


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request_id → bản ghi đang mở (chờ record_output)
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Lưu input + thời điểm bắt đầu. Trả về request_id để nối với output."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": _scrub(text),
            "input_chars": len(text or ""),
            "_t0": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output, quyết định chặn, lớp chặn, latency → append vào self.logs."""
        entry = None
        if request_id and request_id in self._open:
            entry = self._open.pop(request_id)
        else:
            # Không có request_id: lấy bản ghi mở gần nhất của user này
            for rid in reversed(list(self._open)):
                if self._open[rid]["user_id"] == user_id:
                    entry = self._open.pop(rid)
                    break
        if entry is None:  # output không có input tương ứng — vẫn ghi để không mất dấu
            entry = {"request_id": request_id or uuid.uuid4().hex[:12],
                     "user_id": user_id, "timestamp": utc_now_iso(),
                     "input": None, "input_chars": 0, "_t0": None}

        t0 = entry.pop("_t0", None)
        entry.update({
            "output": _scrub(text),
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1) if t0 else None,
        })
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
