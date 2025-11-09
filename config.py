"""全局配置工具。

核心环境变量（可在 ``.env`` 中配置）：

- BASE_DIR, DATA_DIR, STORAGE_DIR, RAG_INDEX_FILE
- REQUEST_TIMEOUT, USER_AGENT
- WEB_SEARCH_API_URL / WEB_SEARCH_API_KEY
- WEATHER_API_URL / WEATHER_API_KEY
- FINANCE_API_URL / FINANCE_API_KEY / FINANCE_PROVIDER
- TRANSPORT_API_URL / TRANSPORT_API_KEY

模块通过 ``Settings`` 数据类对上述变量统一封装，并提供 ``get_settings`` 缓存访问。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_ENV_FILENAMES = (".env",)


def _load_env_file(path: Path) -> None:
    """Populate ``os.environ`` with key/value pairs from ``path`` if present.

    Only lines in ``KEY=VALUE`` form are considered. Existing environment
    variables are never overwritten, making it safe to call multiple times.
    """
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_environment(env_file: Optional[Path | str] = None) -> None:
    """Load environment variables from ``env_file`` or the default candidates."""
    if env_file is not None:
        _load_env_file(Path(env_file))
        return

    for filename in _DEFAULT_ENV_FILENAMES:
        _load_env_file(PROJECT_ROOT / filename)


def _to_path(value: Optional[str], fallback: Path) -> Path:
    return Path(value).expanduser().resolve() if value else fallback


def _to_float(value: Optional[str], fallback: float) -> float:
    try:
        return float(value) if value is not None else fallback
    except ValueError:
        return fallback


@dataclass(frozen=True)
class Settings:
    """Container object holding all runtime configuration values."""

    base_dir: Path
    data_dir: Path
    storage_dir: Path
    rag_index_file: Path
    llama_index_dir: Path
    request_timeout: float

    web_search_api_url: Optional[str]
    web_search_api_key: Optional[str]
    web_search_api_method: str
    web_search_auth_header: str
    web_search_auth_prefix: str

    weather_api_url: str
    weather_api_key: Optional[str]

    finance_api_url: str
    finance_api_key: Optional[str]

    transport_api_url: Optional[str]
    transport_api_key: Optional[str]

    user_agent: str
    llama_embedding_model: Optional[str]
    llama_device: Optional[str]

    deepseek_api_key: Optional[str]
    deepseek_url: str
    deepseek_model: str

    azure_api_key: Optional[str]
    azure_url: str

    @classmethod
    def from_env(cls, env_file: Optional[Path | str] = None) -> "Settings":
        """Build a ``Settings`` instance using environment variables."""
        load_environment(env_file)

        base_dir = _to_path(
            os.environ.get("BASE_DIR"),
            Path(__file__).resolve().parent,
        )
        data_dir = _to_path(os.environ.get("DATA_DIR"), base_dir / "data")
        storage_dir = _to_path(os.environ.get("STORAGE_DIR"), base_dir / "storage")

        rag_index = _to_path(
            os.environ.get("RAG_INDEX_FILE"),
            storage_dir / "local_rag_index.json",
        )
        llama_dir = _to_path(
            os.environ.get("LLAMA_PERSIST_DIR"),
            storage_dir / "llama_index",
        )

        return cls(
            base_dir=base_dir,
            data_dir=data_dir,
            storage_dir=storage_dir,
            rag_index_file=rag_index,
            llama_index_dir=llama_dir,
            request_timeout=_to_float(os.environ.get("REQUEST_TIMEOUT"), 10.0),
            web_search_api_url=os.environ.get("WEB_SEARCH_API_URL"),
            web_search_api_key=os.environ.get("WEB_SEARCH_API_KEY"),
            web_search_api_method=os.environ.get("WEB_SEARCH_API_METHOD", "GET").upper(),
            web_search_auth_header=os.environ.get("WEB_SEARCH_AUTH_HEADER", "Authorization"),
            web_search_auth_prefix=os.environ.get("WEB_SEARCH_AUTH_PREFIX", "Bearer "),
            weather_api_url=os.environ.get("WEATHER_API_URL"),
            weather_api_key=os.environ.get("WEATHER_API_KEY"),
            finance_api_url=os.environ.get("FINANCE_API_URL"),
            finance_api_key=os.environ.get("FINANCE_API_KEY"),
            transport_api_url=os.environ.get("TRANSPORT_API_URL"),
            transport_api_key=os.environ.get("TRANSPORT_API_KEY"),
            user_agent=os.environ.get("USER_AGENT", "IntelligentSearchEngine/1.0"),
            llama_embedding_model=os.environ.get("LLAMA_EMBEDDING_MODEL"),
            llama_device=os.environ.get("LLAMA_DEVICE"),
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
            deepseek_url=os.environ.get("DEEPSEEK_URL"),
            deepseek_model=os.environ.get("DEEPSEEK_MODEL"),
            azure_api_key=os.environ.get("AZURE_API_KEY"),
            azure_url=os.environ.get("AZURE_URL"),
        )

    def ensure_directories(self) -> None:
        """Create storage-related directories if they do not yet exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)


_SETTINGS_CACHE: Optional[Settings] = None


def get_settings(env_file: Optional[Path | str] = None) -> Settings:
    """Return cached ``Settings`` instance, loading it on first access."""
    global _SETTINGS_CACHE

    if _SETTINGS_CACHE is None or env_file is not None:
        _SETTINGS_CACHE = Settings.from_env(env_file)

    return _SETTINGS_CACHE


__all__ = [
    "Settings",
    "get_settings",
    "load_environment",
]
