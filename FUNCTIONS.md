# KDSH 2.0 Function Reference

## Core Pipeline Functions

### final_decision.py
- **aggregate_final_decision()** - Main decision pipeline with multi-stage evaluation
- **is_emotion_motivation_atom()** - Protects emotion/motivation atoms from HARD_VIOLATION
- **classify_atom_semantic_type()** - Classifies atoms as EVENT, RITUAL, EMOTION, etc.
- **count_event_atoms()** - Counts concrete event atoms in claim
- **is_historical_texture_only()** - Checks if all atoms are historical texture types
- **is_named_character_event()** - Detects atoms involving proper names
- **has_canon_confirmation()** - Checks if atom has canon evidence confirmation
- **is_unique_interaction_event()** - Identifies unique timeline-anchoring interactions
- **is_central_event()** - **NEW**: Detects CENTRAL_CANON_EVENT atoms (canon anchors only, hard blacklist for identity)
- **is_causal_anchor()** - **NEW**: Detects CAUSAL_ANCHOR atoms (causal language + canon events)
- **is_identity_or_background()** - **NEW**: Guards against identity/background contradictions
- **involves_two_named_characters()** - **NEW**: Detects two-character interactions (handles hyphenated names)
- **no_canon_confirmation()** - **NEW**: Helper function for canon confirmation checks
- **detect_canon_expected_events()** - CANON_EXPECTED_EVENT Rule for canon anchor events only
- **detect_fabricated_event_anchors()** - Detects fabricated events with no canon footprint
- **get_canon_event_tier()** - Classifies canon events by criticality (Tier 1-3)
- **is_canon_obligated_event()** - Checks if atom requires canon support
- **classify_claim_type()** - Routes claims for ensemble processing
- **compute_salience()** - Determines claim importance (HIGH/LOW)

### claim_decomposer.py
- **decompose_claim()** - Breaks claims into atomic facts using GLM-4.7

### grounded_inference.py
- **grounded_constraint_inference()** - Evaluates atoms against evidence chunks
- Returns: HARD_VIOLATION, UNSUPPORTED, SUPPORTED, NO_CONSTRAINT

### ensemble_v1.py
- **run_ensemble_decision()** - Three-perspective LLM ensemble (ARCHIVIST, HISTORIAN, PROSECUTOR)
- **create_archivist_prompt()** - Evidence-focused perspective
- **create_historian_prompt()** - Timeline consistency perspective  
- **create_prosecutor_prompt()** - Contradiction-seeking perspective
- **aggregate_ensemble_verdicts()** - Deterministic ensemble aggregation

### semantic_index.py
- **SemanticIndex** - E5-large-v2 embeddings with FAISS indexing
- **semantic_retrieve()** - Retrieves top-K relevant chunks
- **add_chunks()** - Adds book chunks to index
- **load_cached_index()** - Loads pre-computed embeddings

### semantic_neighborhood.py
- **semantic_neighborhood_evaluation()** - Narrative compatibility assessment using GLM-4.7

## Data Processing Functions

### chunking.py
- **chunk_book_cdgf()** - C-D-F-G chunking strategy
- **create_sliding_windows()** - Overlapping text windows
- **detect_section_boundaries()** - Chapter/section detection
- **tag_characters()** - Character mention tagging
- **identify_temporal_phases()** - Timeline phase detection

### load_books.py
- **load_book_text()** - Loads and preprocesses book files
- **normalize_book_name()** - Standardizes book naming

### text_normalization.py
- **normalize_text()** - Encoding and formatting consistency
- **clean_unicode()** - Unicode character handling

## Retrieval Functions

### hybrid_retrieval.py
- **hybrid_search()** - Combines semantic + keyword search
- **semantic_search()** - E5-large-v2 vector search
- **keyword_search()** - BM25-style keyword matching

### index_inmemory.py
- **InMemoryIndex** - Keyword indexing for hybrid retrieval
- **build_inverted_index()** - Creates term-document mappings
- **search_keywords()** - Keyword-based chunk retrieval

## Character Analysis Functions

### character_profiles.py
- **generate_character_profile()** - LLM-generated character summaries
- **extract_character_traits()** - Key characteristic extraction
- **compress_character_info()** - Semantic compression for evaluation

### narrative_compatibility.py
- **assess_narrative_fit()** - Evaluates claim compatibility with story
- **check_timeline_consistency()** - Temporal logic validation
- **evaluate_character_consistency()** - Character behavior alignment

## Configuration & Utilities

### config.py
- **MODEL_CONFIG** - GLM-4.7 API configuration
- **EMBEDDING_CONFIG** - E5-large-v2 settings
- **CHUNK_CONFIG** - Chunking parameters
- **RETRIEVAL_CONFIG** - Search parameters

## Key Decision Rules

### Evidence-Provider Model
**Fundamental Principle**: Atoms provide evidence, not decisions. Only specific patterns trigger contradiction:

1. **CAUSAL contradictions**: Unsupported causal explanations for canon events
2. **RELATIONAL contradictions**: Unsupported meetings/interactions between canon characters  
3. **CANON ANCHOR contradictions**: Unsupported creation/alteration of canon events

**BANNED PATTERNS**:
- ❌ Contradiction based on atom count alone
- ❌ CENTRAL_EVENT_UNSUPPORTED on identity/background
- ❌ Any rule firing only because N atoms are unsupported

### Three Core Rules
1. **UNSUPPORTED_CANON_MEETING** - Rejects unconfirmed character meetings
2. **UNSUPPORTED_CAUSAL_ANCHOR** - Rejects unsupported causal explanations  
3. **CANON_ANCHOR_UNSUPPORTED** - Rejects unsupported canon event creation (non-identity only)

### Tiered Canon Event System
- **Tier 3 (Identity Locks)**: Death, imprisonment, first meetings, lineage - 2+ failures → contradict
- **Tier 2 (Timeline Anchors)**: Arrests, wars, marriages - 3+ failures → contradict  
- **Tier 1 (Contextual Canon)**: Memberships, beliefs - never contradict alone

### Canon Expected Event Rule
Triggers contradict when ALL conditions met:
- Atom is EVENT type
- High salience claim
- Unique interaction (meeting, arrest, betrayal)
- Involves named characters
- Zero canon confirmation in evidence
- Requires 2+ violations to trigger

## API Integration

### GLM-4.7 Functions
- **call_glm_api()** - Direct API calls to GLM-4.7
- **handle_rate_limiting()** - API rate management
- **parse_glm_response()** - Response processing

### Error Handling
- **retry_with_backoff()** - Exponential backoff for API failures
- **validate_response()** - Response format validation
- **log_api_errors()** - Error tracking and logging

## Performance Metrics

### Evaluation Functions
- **calculate_accuracy()** - Overall system accuracy
- **compute_precision_recall()** - Precision/recall for CONTRADICT class
- **generate_confusion_matrix()** - Classification breakdown
- **track_method_distribution()** - Decision method usage stats

## Current Performance
- **Accuracy**: ~70% on full dataset
- **Precision (CONTRADICT)**: ~85%
- **Recall (CONTRADICT)**: ~60%
- **F1-Score**: ~70%

## Usage Patterns

### High-Level Flow
1. Load books → Chunk → Embed → Index
2. Receive claim → Retrieve evidence → Decompose atoms
3. Apply Canon Expected Event Rule
4. Evaluate atoms → Apply tiered rules
5. Route to ensemble if needed → Final decision

### Key Thresholds
- Canon Expected Events: ≥2 violations
- Tier 3 Canon Events: ≥2 violations  
- Tier 2 Canon Events: ≥3 violations
- UNSUPPORTED Event Atoms: ≥3 for HIGH salience
- Ensemble Weight: 1.0 for complex cases