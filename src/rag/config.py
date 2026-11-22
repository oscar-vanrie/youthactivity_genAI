"""Paden en instellingen. Alle andere modules halen hier hun waarden vandaan."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


RAW_PATH = BASE_DIR / "data" / "raw" / "games_raw.json"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
EVAL_PATH = BASE_DIR / "data" / "eval" / "eval_set.json"

COLLECTION_NAME = "youth_games"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

CANDIDATE_POOL = 300 
UNKNOWN_AGE = -1
EMPTY_VALUES = {"", "null", "none", "andere", "geen"}


USE_HYBRID = True