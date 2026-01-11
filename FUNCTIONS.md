# KDSH 2.0 Function Reference

## Core Pipeline Functions

### final_decision_ensemble.py
- **aggregate_final_decision()** - Main ensemble decision pipeline with multi-stage evaluation
- **is_canon_obligated_atom()** - Determines if atom requires canon support (obligations only)

### final_decision.py (Legacy - Not Used)
- Contains helper functions but not actively used in production pipeline
- Replaced by ensemble implementation

### claim_decomposer.py
- **decompose_claim()** - Breaks claims into 3-7 atomic facts using Mistral Small 2503
- **ClaimDecomposer** - Class managing decomposition with Mistral API client

### grounded_inference.py
- **infer_grounded_constraint()** - Evaluates atoms against evidence chunks using Mistral Small 2503
- **GroundedInference** - Class managing forensic auditor evaluation
- **High-Stakes Filter**: Requires "competing fact" for HARD_VIOLATION
- **Returns**: HARD_VIOLATION (competing fact exists), UNSUPPORTED (silence), NO_CONSTRAINT, SUPPORTED
- **Validation**: 50% word overlap check to prevent false SUPPORTED verdicts
- **Forensic Auditor Approach**: Silence ≠ Contradiction

### semantic_neighborhood.py
- **evaluate_semantic_compatibility()** - Narrative compatibility assessment using Mistral Small 2503
- **SemanticNeighborhood** - Class managing semantic evaluation with character profiles
- **Returns**: COMPATIBLE, INCOMPATIBLE

## Data Processing Functions

### chunking.py
- **chunk_book_cdgf()** - C-D-F-G chunking strategy
- **create_sliding_windows()** - Overlapping text windows (~850 tokens, 175 overlap)
- **detect_section_boundaries()** - Chapter/section detection
- **tag_characters()** - Lexicon-based character mention tagging
- **identify_temporal_phases()** - Timeline phase detection (early/middle/late)

### load_books.py
- **load_book_text()** - Loads and preprocesses book files with UTF-8 encoding
- **remove_gutenberg_metadata()** - Strips Project Gutenberg headers/footers
- **normalize_book_name()** - Standardizes book naming (lowercase, underscores)

### text_normalization.py
- **normalize_text()** - Encoding and formatting consistency
- **clean_unicode()** - Unicode character handling

## Retrieval Functions

### semantic_index.py
- **SemanticIndex** - E5-large-v2 embeddings with FAISS indexing
- **semantic_retrieve()** - Retrieves top-K relevant chunks (default: 10)
- **add_chunks()** - Adds book chunks to index
- **load_cached_index()** - Loads pre-computed embeddings from cache
- **save_index()** - Saves FAISS index to disk

### hybrid_retrieval.py (Not Used in Production)
- Legacy hybrid search implementation
- Replaced by pure semantic retrieval

### index_inmemory.py (Not Used in Production)
- Legacy keyword indexing
- Replaced by semantic search

## Character Analysis Functions

### character_profiles.py
- **generate_character_profile()** - Mistral-generated character summaries
- **get_or_create_profile()** - Cache-aware profile retrieval
- **collect_character_chunks()** - Gathers all chunks mentioning character
- **hash_chunks()** - Hash-based cache invalidation

### narrative_compatibility.py (Legacy - Not Used)
- Replaced by semantic_neighborhood.py

## Configuration & Utilities

### config.py
- **MODEL_CONFIG** - Mistral API configuration
- **EMBEDDING_CONFIG** - E5-large-v2 settings
- **CHUNK_CONFIG** - Chunking parameters (850 tokens, 175 overlap)
- **RETRIEVAL_CONFIG** - Search parameters (top-k=10)

## Key Decision Rules

### Ensemble Voting System
**Three Perspectives**:
1. **Strict**: Treats NO_CONSTRAINT as UNSUPPORTED (most conservative)
2. **Moderate**: Standard evaluation rules (balanced)
3. **Lenient**: Treats UNSUPPORTED as NO_CONSTRAINT for non-obligations (most permissive)

**Voting Logic**:
- If 2+ perspectives say CONTRADICT → Final verdict: CONTRADICT
- Otherwise → Final verdict: CONSISTENT

### Obligation Detection
**is_canon_obligated_atom()** checks for:
- Obligation keywords: "must", "should", "required", "obligated", "supposed to"
- Returns True if atom contains obligation language

### Strict Support Detection
**Prevents co-occurrence hallucination**:
1. Prompt engineering: Requires explicit statement, not just co-occurrence
2. Validation check: Overrides SUPPORTED if <50% word overlap

## API Integration

### Mistral API Functions
- **Mistral Client**: Direct API calls to mistral-small-2503
- **No Rate Limiting**: Removed for maximum throughput
- **Error Handling**: Try-catch with fallback to rule-based inference

### API Configuration
```python
from mistralai import Mistral
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
response = client.chat.complete(
    model="mistral-small-2503",
    messages=[{"role": "user", "content": prompt}]
)
```

## Performance Metrics

### Evaluation Functions (test_full_clean.py)
- **calculate_accuracy()** - Overall system accuracy
- **compute_precision_recall()** - Precision/recall for CONTRADICT class
- **generate_confusion_matrix()** - Classification breakdown
- **track_method_distribution()** - Decision method usage stats

### Current Performance (30 Random Claims)
- **Accuracy**: 83.33%
- **Precision (CONTRADICT)**: 85.71%
- **Recall (CONTRADICT)**: 80.00%
- **F1-Score**: 82.76%

## Usage Patterns

### High-Level Flow
1. Load books → Chunk (C-D-F-G) → Embed (E5-large-v2) → Index (FAISS)
2. Receive claim → Retrieve evidence (semantic search, top-10)
3. Decompose claim → 3-7 atomic facts (Mistral)
4. Evaluate each atom with 3 perspectives (Mistral)
5. Aggregate votes → Final decision

### API Usage per Claim
- 1 decomposition call
- 3 × num_atoms evaluation calls (strict, moderate, lenient)
- Total: 10-22 requests per claim (for 3-7 atoms)
- For 80 claims: ~800-1,760 total requests

### Cache Utilization
- **Book Chunks**: Loaded from `cache/chunks/` (no API calls)
- **FAISS Embeddings**: Loaded from `cache/embeddings/` (no API calls)
- **Character Profiles**: Loaded from `cache/profiles/` (no API calls)
- **API Calls**: Only for decomposition and atom verification

## Key Thresholds
- **Chunk Size**: ~850 tokens with 175 token overlap
- **Retrieval**: Top-10 chunks per claim
- **Ensemble Voting**: 2+ CONTRADICT votes → CONTRADICT
- **Word Overlap**: 50% minimum for SUPPORTED validation
- **Decomposition**: 3-7 atoms per claim

## Production Files (Active)
- `src/final_decision_ensemble.py` - Main decision logic
- `src/claim_decomposer.py` - Atomic decomposition
- `src/grounded_inference.py` - Evidence evaluation
- `src/semantic_neighborhood.py` - Narrative compatibility
- `src/semantic_index.py` - E5-large-v2 + FAISS
- `src/character_profiles.py` - Character summaries
- `src/chunking.py` - C-D-F-G chunking
- `src/load_books.py` - Book preprocessing
- `test_full_clean.py` - Evaluation script
- `build_cache.py` - Cache generation

## Legacy Files (Not Used)
- `src/final_decision.py` - Replaced by ensemble
- `src/final_decision_backup.py` - Old backup
- `src/final_decision_broken.py` - Old backup
- `src/final_decision_clean.py` - Old backup
- `src/ensemble_v1.py` - Old ensemble implementation
- `src/hybrid_retrieval.py` - Replaced by semantic search
- `src/index_inmemory.py` - Replaced by FAISS
- `src/narrative_compatibility.py` - Replaced by semantic_neighborhood
- All test_*.py files except test_full_clean.py
