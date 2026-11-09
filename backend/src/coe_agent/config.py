"""Paths, model names and source documents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "index" / "vectors.json"

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "backend" / ".env")


def _load_windows_user_env(name: str) -> None:
    """On Windows, `setx` only reaches new processes. Read the user registry as a fallback."""
    if os.environ.get(name) or os.name != "nt":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            if value:
                os.environ[name] = value
    except OSError:
        pass


_load_windows_user_env("GOOGLE_API_KEY")

CHAT_MODEL = os.getenv("COE_CHAT_MODEL", "google_genai:gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("COE_EMBEDDING_MODEL", "models/gemini-embedding-2")

BASE_URL = "https://legislacao.prefeitura.sp.gov.br/leis"


@dataclass(frozen=True)
class Source:
    key: str  # short id used in citations
    kind: str  # "law" (split by article) or "annex" (split by PDF page)
    title: str
    file: str
    url: str


SOURCES: tuple[Source, ...] = (
    Source(
        key="L16642",
        kind="law",
        title="Lei nº 16.642/2017 (Código de Obras e Edificações - COE)",
        file="lei.html",
        url=f"{BASE_URL}/lei-16642-de-09-de-maio-de-2017",
    ),
    Source(
        key="L16642-anexos",
        kind="annex",
        title="Anexos I, II e III da Lei nº 16.642/2017",
        file="lei_anexos.pdf",
        url=f"{BASE_URL}/lei-16642-de-09-de-maio-de-2017/anexo/6965283c30057156f703246c/"
        "Anexos%20I_II%20e%20III%20%20da%20Lei_16642_2017.pdf",
    ),
    Source(
        key="D57776",
        kind="law",
        title="Decreto nº 57.776/2017 (regulamenta o COE)",
        file="dec.html",
        url=f"{BASE_URL}/decreto-57776-de-07-de-julho-de-2017",
    ),
    Source(
        key="D57776-anexos",
        kind="annex",
        title="Anexos do Decreto nº 57.776/2017",
        file="dec_anexos.pdf",
        url=f"{BASE_URL}/decreto-57776-de-07-de-julho-de-2017/anexo/648784ea14119248c7a5a42a/"
        "D57776_2017_anexos.pdf",
    ),
)

SOURCES_BY_KEY = {s.key: s for s in SOURCES}
