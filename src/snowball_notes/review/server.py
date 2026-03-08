from __future__ import annotations

import json
from contextlib import asynccontextmanager
from html import escape

from ..calibrate.confidence_feedback import HUMAN_LABELS, record_confidence_feedback
from ..config import load_config
from ..observability.logger import JsonlLogger
from ..review.cli import approve_review, update_review
from ..storage.sqlite import Database
from ..storage.vault import Vault


def build_review_app(config_path: str | None = None):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("fastapi is required for review serve. Install the review extra first.") from exc

    config = load_config(config_path)
    db = Database(config.db_path)
    db.migrate()
    db.event_logger = JsonlLogger(config.log_path)
    vault = Vault(config)
    @asynccontextmanager
    async def lifespan(_: FastAPI):  # pragma: no cover - framework hook
        try:
            yield
        finally:
            db.close()

    app = FastAPI(title="Snowball Review", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/reviews")
    def list_reviews() -> list[dict]:
        return _pending_reviews(db)

    @app.get("/api/reviews/{review_id}")
    def get_review(review_id: str) -> dict:
        detail = _review_detail(db, review_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"review {review_id} not found")
        return detail

    @app.post("/api/reviews/{review_id}/approve")
    def approve(
        review_id: str,
        payload: dict | None = Body(default=None),
    ) -> dict:
        payload = payload or {}
        approved, detail = approve_review(
            db,
            vault,
            config,
            review_id,
            reviewer=str(payload.get("reviewer") or "web"),
            action=payload.get("action"),
            note_id=payload.get("note_id"),
            title=payload.get("title"),
        )
        if not approved:
            raise HTTPException(status_code=400, detail=detail)
        return {"approved": True, "detail": detail}

    @app.post("/api/reviews/{review_id}/create-separate")
    def create_separate(
        review_id: str,
        payload: dict | None = Body(default=None),
    ) -> dict:
        payload = payload or {}
        approved, detail = approve_review(
            db,
            vault,
            config,
            review_id,
            reviewer=str(payload.get("reviewer") or "web"),
            action="create",
            title=payload.get("title"),
            resolved_as="create_separate",
        )
        if not approved:
            raise HTTPException(status_code=400, detail=detail)
        return {"resolved": True, "final_action": "create_separate", "detail": detail}

    @app.post("/api/reviews/{review_id}/reject")
    def reject(
        review_id: str,
        payload: dict | None = Body(default=None),
    ) -> dict:
        payload = payload or {}
        updated = update_review(db, review_id, "rejected", reviewer=str(payload.get("reviewer") or "web"))
        if not updated:
            raise HTTPException(status_code=404, detail=f"review {review_id} not found")
        db.commit()
        return {"rejected": True}

    @app.post("/api/reviews/{review_id}/mark-conflict")
    def mark_conflict(
        review_id: str,
        payload: dict | None = Body(default=None),
    ) -> dict:
        payload = payload or {}
        review_row = db.fetchone(
            """
            SELECT final_target_note_id, suggested_target_note_id
            FROM review_actions
            WHERE review_id = ?
            """,
            (review_id,),
        )
        if review_row is None:
            raise HTTPException(status_code=404, detail=f"review {review_id} not found")
        final_target_note_id = (
            payload.get("note_id")
            or review_row.get("suggested_target_note_id")
            or review_row.get("final_target_note_id")
        )
        updated = update_review(
            db,
            review_id,
            "mark_conflict",
            reviewer=str(payload.get("reviewer") or "web"),
            final_target_note_id=final_target_note_id,
            reason=str(payload.get("reason") or "marked_conflict"),
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"review {review_id} not found")
        db.commit()
        return {
            "resolved": True,
            "final_action": "mark_conflict",
            "final_target_note_id": final_target_note_id,
        }

    @app.post("/api/reviews/{review_id}/discard")
    def discard(
        review_id: str,
        payload: dict | None = Body(default=None),
    ) -> dict:
        payload = payload or {}
        updated = update_review(
            db,
            review_id,
            "discarded",
            reviewer=str(payload.get("reviewer") or "web"),
            reason=str(payload.get("reason") or "discarded_by_reviewer"),
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"review {review_id} not found")
        db.commit()
        return {"resolved": True, "final_action": "discarded"}

    @app.post("/api/reviews/{review_id}/confidence-feedback")
    def add_feedback(
        review_id: str,
        payload: dict | None = Body(default=None),
    ) -> dict:
        payload = payload or {}
        label = str(payload.get("label") or "")
        if label not in HUMAN_LABELS:
            raise HTTPException(status_code=400, detail=f"label must be one of {sorted(HUMAN_LABELS)}")
        review_row = db.fetchone("SELECT turn_id FROM review_actions WHERE review_id = ?", (review_id,))
        if review_row is None:
            raise HTTPException(status_code=404, detail=f"review {review_id} not found")
        created = record_confidence_feedback(
            db,
            review_row["turn_id"],
            label,
            annotator=str(payload.get("annotator") or "web"),
        )
        if not created:
            raise HTTPException(status_code=404, detail=f"turn {review_row['turn_id']} not found")
        db.commit()
        return {"recorded": True, "turn_id": review_row["turn_id"], "label": label}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _render_index_html()

    return app


def serve_review_app(config_path: str | None = None, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("uvicorn is required for review serve. Install the review extra first.") from exc
    app = build_review_app(config_path)
    uvicorn.run(app, host=host, port=port)


def _pending_reviews(db) -> list[dict]:
    return db.fetchall(
        """
        SELECT review_id, turn_id, trace_id, final_target_note_id, suggested_action,
               suggested_target_note_id, suggested_payload_json, reason, created_at
        FROM review_actions
        WHERE final_action = 'pending_review'
        ORDER BY created_at DESC
        """
    )


def _review_detail(db, review_id: str) -> dict | None:
    review_row = db.fetchone(
        """
        SELECT review_id, turn_id, trace_id, final_action, final_target_note_id,
               suggested_action, suggested_target_note_id, suggested_payload_json,
               reviewer, reason, created_at
        FROM review_actions
        WHERE review_id = ?
        """,
        (review_id,),
    )
    if review_row is None:
        return None
    event_row = db.fetchone(
        """
        SELECT payload_json
        FROM conversation_events
        WHERE turn_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (review_row["turn_id"],),
    )
    trace_row = db.fetchone(
        """
        SELECT trace_id, final_decision, terminal_reason, trace_json, created_at
        FROM agent_traces
        WHERE trace_id = ?
        """,
        (review_row["trace_id"],),
    )
    replay_row = db.fetchone(
        """
        SELECT trace_id, event_json, tool_results_json, knowledge_snapshot_refs_json,
               model_name, model_adapter_version, created_at
        FROM replay_bundles
        WHERE trace_id = ?
        """,
        (review_row["trace_id"],),
    )
    proposals = db.fetchall(
        """
        SELECT proposal_id, action_type, target_note_id, payload_json, status, created_at, committed_at
        FROM action_proposals
        WHERE trace_id = ?
        ORDER BY created_at ASC, proposal_id ASC
        """,
        (review_row["trace_id"],),
    )
    detail = dict(review_row)
    if detail.get("suggested_payload_json"):
        detail["suggested_payload_json"] = json.loads(detail["suggested_payload_json"])
    if event_row:
        detail["event"] = json.loads(event_row["payload_json"])
    if trace_row:
        detail["trace"] = {
            "trace_id": trace_row["trace_id"],
            "final_decision": trace_row["final_decision"],
            "terminal_reason": trace_row["terminal_reason"],
            "trace": json.loads(trace_row["trace_json"]),
            "created_at": trace_row["created_at"],
        }
    if replay_row:
        detail["replay_bundle"] = {
            "trace_id": replay_row["trace_id"],
            "event": json.loads(replay_row["event_json"]),
            "tool_results": json.loads(replay_row["tool_results_json"]),
            "knowledge_snapshot_refs": json.loads(replay_row["knowledge_snapshot_refs_json"]),
            "model_name": replay_row["model_name"],
            "model_adapter_version": replay_row["model_adapter_version"],
            "created_at": replay_row["created_at"],
        }
    detail["proposals"] = [
        {
            **proposal,
            "payload_json": json.loads(proposal["payload_json"]),
        }
        for proposal in proposals
    ]
    return detail


def _render_index_html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Snowball Review</title>
  <style>
    :root {{
      --bg: #efe7d4;
      --panel: rgba(255, 251, 242, 0.92);
      --panel-strong: #fff9ef;
      --line: #d4bf97;
      --ink: #21170f;
      --muted: #725c47;
      --accent: #b54d2f;
      --accent-2: #2f6f62;
      --warn: #7f1d1d;
      --shadow: 0 18px 40px rgba(76, 50, 18, 0.12);
      --mono: "SF Mono", "JetBrains Mono", Menlo, monospace;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(181, 77, 47, 0.16), transparent 28%),
        radial-gradient(circle at bottom right, rgba(47, 111, 98, 0.18), transparent 25%),
        linear-gradient(180deg, #f5efdf 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: var(--serif);
    }}
    .shell {{
      max-width: 1520px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.25fr 0.75fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .hero-card, .panel {{
      background: var(--panel);
      border: 1px solid rgba(164, 133, 89, 0.28);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .hero-card {{
      padding: 24px 26px;
    }}
    .eyebrow {{
      font: 600 11px/1.2 var(--mono);
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent-2);
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(32px, 5vw, 60px);
      line-height: 0.92;
      letter-spacing: -0.04em;
    }}
    .subcopy {{
      margin-top: 14px;
      max-width: 56ch;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.5;
    }}
    .stats {{
      display: grid;
      gap: 12px;
      padding: 22px;
      align-content: start;
    }}
    .stat {{
      display: grid;
      gap: 4px;
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 249, 239, 0.85);
      border: 1px solid rgba(180, 151, 111, 0.28);
    }}
    .stat-label {{
      font: 600 11px/1.2 var(--mono);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .stat-value {{
      font-size: 30px;
      line-height: 1;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
      gap: 18px;
      min-height: 70vh;
    }}
    .sidebar, .detail {{
      padding: 18px;
    }}
    .panel-title {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .panel-title h2 {{
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.03em;
    }}
    .panel-title span {{
      color: var(--muted);
      font: 500 12px/1.2 var(--mono);
    }}
    .review-list {{
      display: grid;
      gap: 10px;
    }}
    .review-item {{
      display: grid;
      gap: 8px;
      width: 100%;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid transparent;
      background: rgba(255, 249, 239, 0.88);
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
      font: inherit;
    }}
    .review-item:hover, .review-item.active {{
      transform: translateY(-1px);
      border-color: rgba(181, 77, 47, 0.44);
      background: #fffdf8;
    }}
    .review-item code {{
      font: 600 12px/1.35 var(--mono);
      color: var(--accent-2);
      word-break: break-all;
    }}
    .review-item .reason {{
      font-size: 15px;
      line-height: 1.35;
    }}
    .review-item .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font: 500 12px/1.35 var(--mono);
    }}
    .detail {{
      display: grid;
      gap: 16px;
      align-content: start;
    }}
    .empty {{
      display: grid;
      place-items: center;
      min-height: 420px;
      border: 1px dashed rgba(164, 133, 89, 0.5);
      border-radius: 20px;
      color: var(--muted);
      background: rgba(255, 251, 242, 0.72);
      text-align: center;
      padding: 24px;
    }}
    .headline {{
      display: grid;
      gap: 8px;
      padding: 18px 20px;
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(181, 77, 47, 0.08), rgba(47, 111, 98, 0.08));
      border: 1px solid rgba(180, 151, 111, 0.28);
    }}
    .headline h3 {{
      margin: 0;
      font-size: clamp(26px, 4vw, 42px);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }}
    .tag-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255, 249, 239, 0.95);
      border: 1px solid rgba(180, 151, 111, 0.34);
      font: 600 12px/1.2 var(--mono);
      color: var(--muted);
    }}
    .columns {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .section {{
      padding: 18px 20px;
      border-radius: 20px;
      background: var(--panel-strong);
      border: 1px solid rgba(180, 151, 111, 0.28);
    }}
    .section h4 {{
      margin: 0 0 10px 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      font-family: var(--mono);
      color: var(--accent-2);
    }}
    .section p, .section li {{
      font-size: 16px;
      line-height: 1.55;
      color: var(--ink);
    }}
    .section ul {{
      margin: 0;
      padding-left: 18px;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 16px;
      background: #1f1c19;
      color: #f6eddc;
      overflow: auto;
      font: 12px/1.55 var(--mono);
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .tools {{
      display: grid;
      gap: 12px;
    }}
    .tool-step {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(240, 232, 214, 0.7);
      border: 1px solid rgba(180, 151, 111, 0.24);
    }}
    .tool-step .tool-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 6px;
      font: 600 12px/1.2 var(--mono);
      color: var(--muted);
    }}
    .tool-step .tool-name {{
      color: var(--accent);
    }}
    .actions {{
      display: grid;
      gap: 14px;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font: 600 11px/1.2 var(--mono);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    input, select, textarea, button {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(180, 151, 111, 0.35);
      padding: 12px 13px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 252, 246, 0.95);
    }}
    textarea {{
      min-height: 108px;
      resize: vertical;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.5;
    }}
    button {{
      cursor: pointer;
      transition: transform 120ms ease, filter 120ms ease;
      font-family: var(--mono);
      font-weight: 600;
    }}
    button:hover {{
      transform: translateY(-1px);
      filter: brightness(0.98);
    }}
    .approve {{
      background: var(--accent-2);
      color: #f4f0e8;
    }}
    .reject {{
      background: #f7d7cc;
      color: var(--warn);
    }}
    .secondary {{
      background: #efe0bc;
      color: var(--ink);
    }}
    .status {{
      min-height: 22px;
      font: 600 12px/1.4 var(--mono);
      color: var(--accent-2);
    }}
    .status.error {{
      color: var(--warn);
    }}
    @media (max-width: 1080px) {{
      .hero, .grid, .columns, .form-grid {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        padding: 18px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <article class="hero-card">
        <div class="eyebrow">Snowball Review Console</div>
        <h1>Resolve flagged runs before they write noise into the vault.</h1>
        <p class="subcopy">
          Pick a pending review, inspect the original turn, trace, proposals, and replay snapshot,
          then approve, reject, or attach parser confidence feedback without leaving the page.
        </p>
      </article>
      <aside class="hero-card stats">
        <div class="stat">
          <div class="stat-label">Pending Reviews</div>
          <div class="stat-value" id="pending-count">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Selected Review</div>
          <div class="stat-value" id="selected-review">-</div>
        </div>
        <div class="stat">
          <div class="stat-label">Health</div>
          <div class="stat-value" id="health-state">Loading</div>
        </div>
      </aside>
    </section>

    <section class="grid">
      <aside class="panel sidebar">
        <div class="panel-title">
          <h2>Queue</h2>
          <span id="queue-meta">pending_review</span>
        </div>
        <div class="review-list" id="review-list"></div>
      </aside>

      <main class="panel detail" id="detail-root">
        <div class="empty">
          <div>
            <div class="eyebrow">No Selection</div>
            <p>Select a review from the left to inspect the run and take action.</p>
          </div>
        </div>
      </main>
    </section>
  </div>

  <script>
    const HUMAN_LABEL_OPTIONS = {json.dumps(sorted(HUMAN_LABELS), ensure_ascii=False)};
    const ACTION_OPTIONS = [
      {{ value: "create", label: "Approve Create" }},
      {{ value: "append", label: "Approve Append" }},
      {{ value: "archive", label: "Approve Archive" }},
      {{ value: "link", label: "Approve Link" }},
    ];

    const state = {{
      reviews: [],
      selectedReviewId: null,
      selectedDetail: null,
    }};

    async function request(path, options = {{}}) {{
      const response = await fetch(path, {{
        headers: {{ "Content-Type": "application/json" }},
        ...options,
      }});
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : await response.text();
      if (!response.ok) {{
        const detail = typeof payload === "object" && payload ? (payload.detail || JSON.stringify(payload)) : String(payload);
        throw new Error(detail);
      }}
      return payload;
    }}

    function pretty(value) {{
      return JSON.stringify(value, null, 2);
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function normalizeActionValue(action) {{
      if (action === "create_note") return "create";
      if (action === "append_note") return "append";
      if (action === "archive_turn") return "archive";
      if (action === "link_notes") return "link";
      return action || "create";
    }}

    function renderReviewList() {{
      const root = document.getElementById("review-list");
      document.getElementById("pending-count").textContent = String(state.reviews.length);
      document.getElementById("selected-review").textContent = state.selectedReviewId || "-";
      root.innerHTML = "";
      if (!state.reviews.length) {{
        root.innerHTML = `
          <div class="empty">
            <div>
              <div class="eyebrow">Queue Empty</div>
              <p>No pending reviews remain.</p>
            </div>
          </div>
        `;
        return;
      }}
      for (const review of state.reviews) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "review-item" + (review.review_id === state.selectedReviewId ? " active" : "");
        button.innerHTML = `
          <code>${{escapeHtml(review.review_id)}}</code>
          <div class="reason">${{escapeHtml(review.reason || "No reason recorded.")}}</div>
          <div class="meta">
            <span>${{escapeHtml(review.turn_id)}}</span>
            <span>${{escapeHtml(review.suggested_action || "create_note")}}</span>
            <span>${{escapeHtml(review.suggested_target_note_id || "-")}}</span>
          </div>
        `;
        button.addEventListener("click", () => selectReview(review.review_id));
        root.appendChild(button);
      }}
    }}

    function renderDetail() {{
      const root = document.getElementById("detail-root");
      const detail = state.selectedDetail;
      if (!detail) {{
        root.innerHTML = `
          <div class="empty">
            <div>
              <div class="eyebrow">No Selection</div>
              <p>Select a review from the left to inspect the run and take action.</p>
            </div>
          </div>
        `;
        return;
      }}

      const event = detail.event || {{}};
      const trace = detail.trace || {{}};
      const traceData = trace.trace || {{}};
      const replay = detail.replay_bundle || {{}};
      const toolSteps = (traceData.steps || []).filter((step) => step.tool_name);
      const actionValue = normalizeActionValue(detail.suggested_action || "create_note");
      const actionOptions = ACTION_OPTIONS.map((option) => `
        <option value="${{option.value}}"${{option.value === actionValue ? " selected" : ""}}>${{option.label}}</option>
      `).join("");
      const labelOptions = HUMAN_LABEL_OPTIONS.map((value) => `
        <option value="${{escapeHtml(value)}}">${{escapeHtml(value)}}</option>
      `).join("");
      const proposalItems = (detail.proposals || []).map((proposal) => `
        <li>
          <strong>${{escapeHtml(proposal.action_type)}}</strong>
          target=${{escapeHtml(proposal.target_note_id || "-")}}
          status=${{escapeHtml(proposal.status)}}
        </li>
      `).join("") || "<li>No proposals were stored for this trace.</li>";
      const toolItems = toolSteps.map((step) => `
        <div class="tool-step">
          <div class="tool-head">
            <span class="tool-name">${{escapeHtml(step.tool_name)}}</span>
            <span>step ${{escapeHtml(step.step_index)}}</span>
          </div>
          <div class="tool-head">
            <span>${{escapeHtml(step.runtime_state)}}</span>
            <span>${{step.guardrail_blocked ? "guardrail blocked" : (step.tool_success ? "success" : "error")}}</span>
          </div>
          <pre>${{escapeHtml(step.tool_input_json || "")}}</pre>
          <pre>${{escapeHtml(step.tool_result_json || "")}}</pre>
        </div>
      `).join("") || "<p>No tool calls recorded.</p>";

      root.innerHTML = `
        <section class="headline">
          <div class="eyebrow">Pending Review</div>
          <h3>${{escapeHtml(detail.reason || "Review")}}</h3>
          <div class="tag-row">
            <span class="pill">${{escapeHtml(detail.review_id)}}</span>
            <span class="pill">turn ${{escapeHtml(detail.turn_id)}}</span>
            <span class="pill">trace ${{escapeHtml(detail.trace_id)}}</span>
            <span class="pill">${{escapeHtml(detail.suggested_action || "create_note")}}</span>
            <span class="pill">target ${{escapeHtml(detail.suggested_target_note_id || detail.final_target_note_id || "-")}}</span>
          </div>
        </section>

        <section class="actions section">
          <h4>Resolve Review</h4>
          <div class="form-grid">
            <label>
              Action
              <select id="approve-action">${{actionOptions}}</select>
            </label>
            <label>
              Reviewer
              <input id="approve-reviewer" value="web">
            </label>
            <label>
              Note ID
              <input id="approve-note-id" value="${{escapeHtml(detail.suggested_target_note_id || detail.final_target_note_id || "")}}">
            </label>
            <label>
              Title
              <input id="approve-title" value="${{escapeHtml(detail.suggested_payload_json?.title || "")}}">
            </label>
          </div>
          <div class="form-grid">
            <button class="approve" id="approve-button" type="button">Approve</button>
            <button class="reject" id="reject-button" type="button">Reject</button>
          </div>
          <div class="form-grid">
            <button class="secondary" id="separate-button" type="button">Create Separate</button>
            <button class="secondary" id="conflict-button" type="button">Mark Conflict</button>
          </div>
          <button class="secondary" id="discard-button" type="button">Discard</button>
          <div class="form-grid">
            <label>
              Confidence Label
              <select id="feedback-label">${{labelOptions}}</select>
            </label>
            <label>
              Feedback Annotator
              <input id="feedback-reviewer" value="web">
            </label>
          </div>
          <button class="secondary" id="feedback-button" type="button">Record Confidence Feedback</button>
          <div class="status" id="action-status"></div>
        </section>

        <section class="columns">
          <article class="section">
            <h4>User Message</h4>
            <p>${{escapeHtml(event.user_message || "")}}</p>
          </article>
          <article class="section">
            <h4>Assistant Final Answer</h4>
            <p>${{escapeHtml(event.assistant_final_answer || "")}}</p>
          </article>
        </section>

        <section class="columns">
          <article class="section">
            <h4>Trace Summary</h4>
            <ul>
              <li>final_decision: ${{escapeHtml(trace.final_decision || "-")}}</li>
              <li>terminal_reason: ${{escapeHtml(trace.terminal_reason || "-")}}</li>
              <li>steps: ${{escapeHtml(traceData.total_steps || 0)}}</li>
              <li>confidence: ${{escapeHtml(traceData.final_confidence || event.source_confidence || "-")}}</li>
            </ul>
          </article>
          <article class="section">
            <h4>Proposals</h4>
            <ul>${{proposalItems}}</ul>
          </article>
        </section>

        <section class="section">
          <h4>Tool Trace</h4>
          <div class="tools">${{toolItems}}</div>
        </section>

        <section class="columns">
          <article class="section">
            <h4>Replay Snapshot</h4>
            <pre>${{escapeHtml(pretty(replay.tool_results || []))}}</pre>
          </article>
          <article class="section">
            <h4>Knowledge Snapshot</h4>
            <pre>${{escapeHtml(pretty(replay.knowledge_snapshot_refs || []))}}</pre>
          </article>
        </section>

        <section class="section">
          <h4>Raw Detail</h4>
          <pre>${{escapeHtml(pretty(detail))}}</pre>
        </section>
      `;

      document.getElementById("approve-button").addEventListener("click", approveSelected);
      document.getElementById("reject-button").addEventListener("click", rejectSelected);
      document.getElementById("separate-button").addEventListener("click", createSeparateSelected);
      document.getElementById("conflict-button").addEventListener("click", markConflictSelected);
      document.getElementById("discard-button").addEventListener("click", discardSelected);
      document.getElementById("feedback-button").addEventListener("click", recordFeedback);
    }}

    function setStatus(message, isError = false) {{
      const node = document.getElementById("action-status");
      if (!node) {{
        return;
      }}
      node.textContent = message;
      node.classList.toggle("error", isError);
    }}

    async function loadReviews(preserveSelection = true) {{
      state.reviews = await request("/api/reviews");
      const stillExists = preserveSelection && state.reviews.some((item) => item.review_id === state.selectedReviewId);
      if (!stillExists) {{
        state.selectedReviewId = state.reviews[0]?.review_id || null;
      }}
      renderReviewList();
      if (state.selectedReviewId) {{
        await selectReview(state.selectedReviewId, false);
      }} else {{
        state.selectedDetail = null;
        renderDetail();
      }}
    }}

    async function selectReview(reviewId, rerenderList = true) {{
      state.selectedReviewId = reviewId;
      if (rerenderList) {{
        renderReviewList();
      }}
      state.selectedDetail = await request(`/api/reviews/${{encodeURIComponent(reviewId)}}`);
      renderDetail();
    }}

    async function approveSelected() {{
      if (!state.selectedReviewId) {{
        return;
      }}
      const payload = {{
        action: document.getElementById("approve-action").value,
        reviewer: document.getElementById("approve-reviewer").value,
        note_id: document.getElementById("approve-note-id").value || null,
        title: document.getElementById("approve-title").value || null,
      }};
      try {{
        const result = await request(`/api/reviews/${{encodeURIComponent(state.selectedReviewId)}}/approve`, {{
          method: "POST",
          body: JSON.stringify(payload),
        }});
        setStatus(`Approved: ${{result.detail}}`);
        await loadReviews(false);
      }} catch (error) {{
        setStatus(error.message, true);
      }}
    }}

    async function rejectSelected() {{
      if (!state.selectedReviewId) {{
        return;
      }}
      try {{
        await request(`/api/reviews/${{encodeURIComponent(state.selectedReviewId)}}/reject`, {{
          method: "POST",
          body: JSON.stringify({{ reviewer: document.getElementById("approve-reviewer").value || "web" }}),
        }});
        setStatus("Rejected.");
        await loadReviews(false);
      }} catch (error) {{
        setStatus(error.message, true);
      }}
    }}

    async function createSeparateSelected() {{
      if (!state.selectedReviewId) {{
        return;
      }}
      const payload = {{
        reviewer: document.getElementById("approve-reviewer").value || "web",
        title: document.getElementById("approve-title").value || null,
      }};
      try {{
        const result = await request(`/api/reviews/${{encodeURIComponent(state.selectedReviewId)}}/create-separate`, {{
          method: "POST",
          body: JSON.stringify(payload),
        }});
        setStatus(`Resolved as ${{result.final_action}}: ${{result.detail}}`);
        await loadReviews(false);
      }} catch (error) {{
        setStatus(error.message, true);
      }}
    }}

    async function markConflictSelected() {{
      if (!state.selectedReviewId) {{
        return;
      }}
      const payload = {{
        reviewer: document.getElementById("approve-reviewer").value || "web",
        note_id: document.getElementById("approve-note-id").value || null,
      }};
      try {{
        const result = await request(`/api/reviews/${{encodeURIComponent(state.selectedReviewId)}}/mark-conflict`, {{
          method: "POST",
          body: JSON.stringify(payload),
        }});
        setStatus(`Resolved as ${{result.final_action}}.`);
        await loadReviews(false);
      }} catch (error) {{
        setStatus(error.message, true);
      }}
    }}

    async function discardSelected() {{
      if (!state.selectedReviewId) {{
        return;
      }}
      try {{
        const result = await request(`/api/reviews/${{encodeURIComponent(state.selectedReviewId)}}/discard`, {{
          method: "POST",
          body: JSON.stringify({{ reviewer: document.getElementById("approve-reviewer").value || "web" }}),
        }});
        setStatus(`Resolved as ${{result.final_action}}.`);
        await loadReviews(false);
      }} catch (error) {{
        setStatus(error.message, true);
      }}
    }}

    async function recordFeedback() {{
      if (!state.selectedReviewId) {{
        return;
      }}
      const payload = {{
        label: document.getElementById("feedback-label").value,
        annotator: document.getElementById("feedback-reviewer").value || "web",
      }};
      try {{
        const result = await request(`/api/reviews/${{encodeURIComponent(state.selectedReviewId)}}/confidence-feedback`, {{
          method: "POST",
          body: JSON.stringify(payload),
        }});
        setStatus(`Recorded confidence feedback: ${{result.label}}`);
      }} catch (error) {{
        setStatus(error.message, true);
      }}
    }}

    async function boot() {{
      try {{
        await request("/health");
        document.getElementById("health-state").textContent = "Ready";
        await loadReviews();
      }} catch (error) {{
        document.getElementById("health-state").textContent = "Error";
        document.getElementById("detail-root").innerHTML = `
          <div class="empty">
            <div>
              <div class="eyebrow">Boot Failure</div>
              <p>${{escapeHtml(error.message)}}</p>
            </div>
          </div>
        `;
      }}
    }}

    boot();
  </script>
</body>
</html>"""
