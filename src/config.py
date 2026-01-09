from pathlib import Path

# ===== Project root =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ===== Data paths =====
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BOOKS_DIR = RAW_DATA_DIR / "books"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# ===== Chunking config =====
TARGET_TOKENS_PER_CHUNK = 850

# ===== Retrieval config =====
DEFAULT_TOP_K = 5

# ===== Semantic search config (local embeddings only) =====
EMBEDDING_MODEL = "intfloat/e5-large-v2"
SEMANTIC_TOP_K = 5
KEYWORD_TOP_K = 5