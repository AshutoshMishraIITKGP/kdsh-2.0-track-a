# Complete End-to-End Pipeline - KDSH 2.0 Track-A

## Overview
This document describes the complete pipeline from raw books to final narrative consistency verification results.

## Pipeline Architecture

```
Raw Books → Chunking → Embedding → Indexing → Claim Processing → Decision → Results
```

## Stage 1: Data Preparation & Caching

### 1.1 Book Processing & Chunking
**File**: `src/chunking.py`
**Input**: Raw text files from `data/raw/books/`
**Output**: Cached chunks in `cache/chunks/{book_name}.jsonl`

```python
# Run chunking for all books
python build_cache.py
```

**Process**:
- Load books with UTF-8 encoding and BOM handling
- Remove Project Gutenberg metadata (headers/footers)
- Apply C-D-F-G chunking strategy:
  - **C**: Sliding/overlapping chunks (~850 tokens, 175 token overlap)
  - **D**: Section-aware (chapter boundaries)
  - **F**: Character-aware tagging (lexicon-based from train.csv)
  - **G**: Temporal phases (early/middle/late based on position)

**Output Format**:
```json
{
  "chunk_id": "monte_cristo_001",
  "book_id": "the_count_of_monte_cristo", 
  "text": "Chapter 1. Marseilles--The Arrival...",
  "relative_position": 0.05,
  "phase": "early",
  "mentioned_characters": ["edmund_dantes", "fernand_mondego"]
}
```

### 1.2 Embedding Generation
**File**: `src/semantic_index.py`
**Input**: Cached chunks
**Output**: FAISS indices in `cache/embeddings/e5-large-v2/`

**Process**:
- Use `intfloat/e5-large-v2` model with GPU acceleration
- Apply instruction prefixes: `"passage: {text}"` for chunks
- Generate embeddings in batches for efficiency
- Store as FAISS indices for fast similarity search

### 1.3 Character Profile Generation
**File**: `src/character_profiles.py`
**Input**: Character-tagged chunks
**Output**: Profiles in `cache/profiles/{book_name}/{character_name}.txt`

**Process**:
- Collect all chunks mentioning specific character
- Generate LLM-based character profile using Groq/Llama-3.1-8b-instant
- Cache profiles with hash-based invalidation
- Structured format: Role, Traits, Narrative function, etc.

## Stage 2: Claim Processing Pipeline

### 2.1 Claim Loading
**File**: `test_full_clean.py`
**Input**: `data/train.csv` or `data/test.csv`
**Output**: Structured claim objects

```python
{
  'claim_id': 'train_001',
  'book_name': 'the_count_of_monte_cristo',
  'character': 'edmund_dantes', 
  'claim_text': 'Edmund Dantes admired Napoleon Bonaparte...',
  'true_label': 'CONTRADICT'  # Only for training data
}
```

### 2.2 Evidence Retrieval
**File**: `src/semantic_index.py` → `semantic_retrieve()`
**Input**: Claim object
**Output**: Top-K relevant chunks

**Process**:
- Generate query embedding: `"query: {claim_text}"`
- Perform FAISS similarity search
- Filter by character mentions (if character found in book)
- Return top 5 most relevant chunks with similarity scores

### 2.3 Multi-Stage Decision Process
**File**: `src/final_decision.py` → `aggregate_final_decision()`

#### Stage 2.3.1: OVER_SPECIFIED Detection
**Purpose**: Catch dataset traps (invented rituals, secret societies)
**Logic**: Check for fabricated ceremonial/organizational details
**Output**: `CONTRADICT` if over-specified, continue otherwise

#### Stage 2.3.2: Atomic Claim Decomposition  
**File**: `src/claim_decomposer.py`
**Process**: Break complex claims into 3-7 atomic facts
**Example**: 
- Original: "Edmund admired Napoleon and joined secret society"
- Atoms: ["Edmund admired Napoleon", "Edmund joined secret society"]

#### Stage 2.3.3: Grounded Inference
**File**: `src/grounded_inference.py`
**Input**: Individual atoms + evidence chunks
**Output**: `HARD_VIOLATION` | `UNSUPPORTED` | `NO_CONSTRAINT`

**Classification Rules**:
- **HARD_VIOLATION**: Explicit text contradictions only
- **UNSUPPORTED**: Introduces detailed new facts not in text  
- **NO_CONSTRAINT**: Consistent with available evidence

#### Stage 2.3.4: Impact Classification & Routing
**Logic**: Determine if UNSUPPORTED atoms need semantic evaluation
- **LOW_IMPACT**: Descriptive details (childhood, rituals) → Skip semantic
- **CAUSAL_IMPACT**: Motivations, alliances, actions → Allow semantic

#### Stage 2.3.5: Semantic Neighborhood Evaluation
**File**: `src/semantic_neighborhood.py`
**Input**: Causal impact atoms + character profile
**Output**: `COMPATIBLE` | `INCOMPATIBLE`

**Process**:
- Use character profile instead of raw chunks
- Evaluate narrative compatibility without requiring explicit evidence
- Apply strict narrative fidelity rules (not just plausibility)

### 2.4 Final Decision Aggregation
**Decision Hierarchy**:
1. `OVER_SPECIFIED` → `CONTRADICT`
2. `HARD_VIOLATION` → `CONTRADICT` 
3. `UNSUPPORTED + LOW_IMPACT` → `CONSISTENT`
4. `UNSUPPORTED + CAUSAL_IMPACT` → Semantic evaluation result
5. `NO_CONSTRAINT` → `CONSISTENT`

## Stage 3: Evaluation & Results

### 3.1 Performance Metrics
**File**: `test_full_clean.py`
**Metrics Calculated**:
- Accuracy: `correct_predictions / total_predictions`
- Precision: `true_positives / (true_positives + false_positives)`
- Recall: `true_positives / (true_positives + false_negatives)`
- F1-Score: `2 * (precision * recall) / (precision + recall)`

### 3.2 Confusion Matrix
```
                 Predicted
                 CONTRADICT  CONSISTENT
Actual CONTRADICT    TP        FN
       CONSISTENT    FP        TN
```

### 3.3 Method Distribution Tracking
- **Grounded decisions**: Resolved without semantic evaluation
- **Semantic-only decisions**: Required narrative compatibility evaluation

## Complete Pipeline Execution

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "GROQ_API_KEY=your_api_key_here" > .env

# Ensure data files exist
# data/train.csv, data/test.csv
# data/raw/books/the_count_of_monte_cristo.txt
# data/raw/books/in_search_of_the_castaways.txt
```

### Step 1: Build Cache (One-time setup)
```bash
python build_cache.py
```
**Output**: 
- `cache/chunks/` - Processed book chunks
- `cache/embeddings/` - FAISS indices  
- `cache/profiles/` - Character profiles

### Step 2: Run Evaluation
```bash
python test_full_clean.py
```

**Sample Output**:
```
=== Full Flow: Grounded + Semantic (30 Random Claims) ===

Loaded 80 total claims
Loading semantic index...
Loaded 526 chunks for the_count_of_monte_cristo
Loaded 524 chunks for in_search_of_the_castaways
Testing 30 random claims...

--- Claim 1: ID train_001 ---
Character: edmund_dantes
Claim: Edmund Dantes admired Napoleon Bonaparte and believed...
True label: CONTRADICT
Retrieved: 5 chunks
Predicted: CONTRADICT
Method: GROUNDED_HARD_VIOLATION
Grounded: HARD_VIOLATION
Semantic: N/A
Atoms: 2, Violations: 1
Correct: True

...

=== RESULTS ===
Correct: 25/30
Accuracy: 83.33%
Precision (CONTRADICT): 85.71%
Recall (CONTRADICT): 80.00%
F1-Score: 82.76%

Confusion Matrix:
                 Predicted
                 CONTRADICT  CONSISTENT
Actual CONTRADICT    12         3
       CONSISTENT     2        13

Method Distribution:
Grounded decisions: 22
Semantic-only decisions: 8
```

## Key Files in Pipeline

### Core Processing
- `src/chunking.py` - C-D-F-G chunking strategy
- `src/semantic_index.py` - E5-large-v2 embeddings + FAISS
- `src/final_decision.py` - Multi-stage decision logic

### Decision Components  
- `src/claim_decomposer.py` - Atomic claim decomposition
- `src/grounded_inference.py` - Evidence-based evaluation
- `src/semantic_neighborhood.py` - Narrative compatibility

### Support Modules
- `src/character_profiles.py` - LLM-generated character summaries
- `src/text_normalization.py` - Encoding consistency
- `build_cache.py` - Cache generation script
- `test_full_clean.py` - Evaluation framework

## Pipeline Performance

### Caching Benefits
- **First run**: ~10-15 minutes (embedding generation)
- **Subsequent runs**: ~30-60 seconds (cached embeddings)
- **Profile generation**: ~2-3 seconds per character (cached)

### API Usage
- **Groq API calls**: ~2-4 per claim (decomposition + grounded inference)
- **Rate limiting**: 2-3 second delays between calls
- **Fallback**: Rule-based inference if API fails

### Memory Requirements
- **Embeddings**: ~100MB per book (E5-large-v2)
- **FAISS indices**: ~50MB per book
- **Runtime memory**: ~2-4GB (GPU recommended for embeddings)

## Architecture Compliance

### Non-Negotiable Constraints ✅
- Chunking logic unchanged (C-D-F-G preserved)
- Grounded verification uses raw chunks and evidence
- Evidence integrity maintained (profiles never cited as evidence)
- Epistemic separation between grounded and semantic evaluation
- Contradiction priority (INCOMPATIBLE overrides COMPATIBLE)

### Enhanced Capabilities ✅  
- ABSENT claims now evaluable for narrative plausibility
- Character profiles provide semantic compression
- E5-large-v2 embeddings for narrative-faithful retrieval
- Transparency with dual verdicts (grounded + semantic)
- Comprehensive caching for fast iteration