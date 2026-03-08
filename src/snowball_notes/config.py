from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

from .utils import parse_scalar


@dataclass
class PathConfig:
    db: str = "./data/snowball.db"
    log: str = "./logs/snowball.jsonl"


@dataclass
class VaultConfig:
    path: str = "./vault"
    inbox_dir: str = "Inbox"
    archive_dir: str = "Archive/Conversations"
    atomic_dir: str = "Knowledge/Atomic"


@dataclass
class IntakeConfig:
    mode: str = "transcript_poll"
    transcript_dir: str = "~/.codex/sessions"
    cli_wrap_file: str | None = None
    parser_version: str = "v1"
    min_response_length: int = 120
    min_confidence_to_run: float = 0.5


@dataclass
class AgentConfig:
    provider: str = "heuristic"
    model: str = "heuristic-v1"
    max_steps: int = 8
    prompt_version: str = "agent_system/v1.md"
    max_writes_per_run: int = 1
    max_appends_per_run: int = 1
    max_model_retries: int = 3
    request_timeout_seconds: int = 60
    api_key_env: str = "OPENAI_API_KEY"
    api_base_url: str = "https://api.openai.com/v1/responses"
    reasoning_effort: str = "medium"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    append_threshold: float = 0.82
    review_threshold: float = 0.62
    title_match_threshold: float = 0.85
    tag_min_overlap: int = 2
    embedding_top_k: int = 5
    embedding_threshold: float = 0.80


@dataclass
class EmbeddingConfig:
    provider: str = "local"
    dashscope_model: str = "text-embedding-v4"
    dashscope_dimensions: int = 1024
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"
    dashscope_api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    voyage_model: str = "voyage-3-lite"
    local_model: str = "hash-384"
    vector_store: str = "sqlite_blob"
    index_text_strategy: str = "title_plus_summary"
    api_key_env: str = "VOYAGE_API_KEY"
    api_base_url: str = "https://api.voyageai.com/v1/embeddings"
    local_dimensions: int = 384


@dataclass
class GuardrailConfig:
    min_confidence_for_note: float = 0.70
    min_confidence_for_append: float = 0.85


@dataclass
class WorkerConfig:
    poll_interval_seconds: int = 10
    claim_timeout_seconds: int = 300
    max_retries: int = 3


@dataclass
class ReconcileConfig:
    enabled: bool = True
    run_on_startup: bool = True
    schedule_cron: str = "0 3 * * *"


@dataclass
class SnowballConfig:
    project_root: Path
    paths: PathConfig = field(default_factory=PathConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)
    intake: IntakeConfig = field(default_factory=IntakeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    reconcile: ReconcileConfig = field(default_factory=ReconcileConfig)

    @property
    def db_path(self) -> Path:
        return resolve_path(self.project_root, self.paths.db)

    @property
    def log_path(self) -> Path:
        return resolve_path(self.project_root, self.paths.log)

    @property
    def vault_path(self) -> Path:
        return resolve_path(self.project_root, self.vault.path)

    @property
    def transcript_dir(self) -> Path:
        return resolve_path(self.project_root, self.intake.transcript_dir)

    @property
    def cli_wrap_path(self) -> Path | None:
        if not self.intake.cli_wrap_file:
            return None
        return resolve_path(self.project_root, self.intake.cli_wrap_file)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": vars(self.paths),
            "vault": vars(self.vault),
            "intake": vars(self.intake),
            "agent": vars(self.agent),
            "retrieval": vars(self.retrieval),
            "embedding": vars(self.embedding),
            "guardrails": vars(self.guardrails),
            "worker": vars(self.worker),
            "reconcile": vars(self.reconcile),
        }


def resolve_path(project_root: Path, value: str) -> Path:
    expanded = Path(os.path.expanduser(value))
    if expanded.is_absolute():
        return expanded
    return (project_root / expanded).resolve()


def default_config(project_root: Path) -> SnowballConfig:
    return SnowballConfig(project_root=project_root.resolve())


def load_config(path: str | Path | None = None) -> SnowballConfig:
    if path is None:
        env_path = os.environ.get("SNOWBALL_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            path = Path.cwd() / "config.yaml"
    config_path = Path(path).resolve()
    project_root = config_path.parent
    config = default_config(project_root)
    if not config_path.exists():
        return config
    data = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    _apply_mapping(config, data)
    return config


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in content.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value == "":
            nested: dict[str, Any] = {}
            current[key] = nested
            stack.append((indent, nested))
            continue
        current[key] = parse_scalar(value)
    return root


def _apply_mapping(config: SnowballConfig, data: dict[str, Any]) -> None:
    for section_name, values in data.items():
        if section_name == "project_root":
            continue
        target = getattr(config, section_name, None)
        if target is None:
            continue
        if isinstance(values, dict):
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
