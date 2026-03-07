from __future__ import annotations

from ..config import SnowballConfig
from ..models import GuardrailResult


ACTION_TOOLS = {
    "propose_create_note",
    "propose_append_to_note",
    "propose_archive_turn",
}
NOTE_CREATION_TOOLS = {"propose_create_note"}


def check_guardrail(config: SnowballConfig, state, tool_name: str) -> GuardrailResult:
    event = state.event
    if tool_name in ACTION_TOOLS and state.write_count >= config.agent.max_writes_per_run:
        return GuardrailResult.block(
            f"write limit exceeded ({config.agent.max_writes_per_run})"
        )
    if tool_name in NOTE_CREATION_TOOLS and event.source_confidence < config.guardrails.min_confidence_for_note:
        return GuardrailResult.block(
            f"source_confidence={event.source_confidence} < {config.guardrails.min_confidence_for_note}"
        )
    if tool_name == "propose_append_to_note":
        if state.append_count >= config.agent.max_appends_per_run:
            return GuardrailResult.block(
                f"append limit exceeded ({config.agent.max_appends_per_run})"
            )
        if event.source_confidence < config.guardrails.min_confidence_for_append:
            return GuardrailResult.block(
                f"append confidence too low ({event.source_confidence})"
            )
    return GuardrailResult.allow()

