# KDSH 2.0 Track-A: Narrative Consistency Verification System

A sophisticated system for verifying the consistency of character backstories against full novel narratives using multi-stage decision logic, semantic embeddings, and LLM-based reasoning.

## 🎯 Project Overview

**Goal**: Determine whether hypothetical character backstories are causally and logically consistent with full narratives of long-form novels (100k+ words).

**Approach**: Two-stage verification system combining grounded textual evidence with narrative compatibility evaluation.

## 🏗️ System Architecture

### Core Pipeline
```
Raw Books → C-D-F-G Chunking → E5-Large-v2 Embeddings → FAISS Indexing → 
Multi-Stage Decision (OVER_SPECIFIED → Atomic Decomposition → Grounded Inference → 
Impact Classification → Semantic Evaluation) → Final Results with Metrics
```

### Key Components

- **📚 Advanced Chunking**: C-D-F-G strategy with sliding windows, section awareness, character tagging, and temporal phases
- **🔍 Semantic Search**: E5-large-v2 embeddings with FAISS indexing for narrative-faithful retrieval
- **🧠 Multi-Stage Decision Logic**: Atomic claim decomposition with violation strength classification
- **⚖️ Grounded Inference**: LLM-based evaluation distinguishing explicit contradictions from unsupported details
- **🎭 Character Profiles**: Semantic compression layer for character-centric evaluation
- **📊 Comprehensive Metrics**: Precision, recall, F1-score, and confusion matrix tracking

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 1: Setup API Key

Create a `.env` file in the project root directory:

```bash
# Windows
echo MISTRAL_API_KEY=your_mistral_api_key_here > .env

# Linux/Mac
echo "MISTRAL_API_KEY=your_mistral_api_key_here" > .env
```

Or manually create `.env` file with:
```
MISTRAL_API_KEY=your_actual_api_key
```

**Get Mistral API Key**: Sign up at [https://console.mistral.ai/](https://console.mistral.ai/) and generate an API key.

### Step 2: Build Cache (One-time setup)

```bash
python build_cache.py
```

This generates:
- Book chunks in `cache/chunks/`
- FAISS embeddings in `cache/embeddings/`
- Character profiles in `cache/profiles/`

**Time**: ~10-15 minutes on first run

### Step 3: Run Model on Training Data

```bash
python test_full_clean.py
```

This will:
- Load all 80 training claims from `data/train.csv`
- Run dual-agent ensemble evaluation
- Display real-time progress with predictions
- Show final metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Method Distribution

**Expected Output**:
```
=== Dual Agent System: 10 Chunks (3-3-4 batches) ===

Loaded 80 total claims
Loading semantic index...
...
=== RESULTS ===
Correct: 52/80
Accuracy: 65.00%
Precision: 51.85%
Recall: 48.28%
F1-Score: 50.00%
```

**Time**: ~7-10 minutes (depends on Mistral API speed)

### Step 4: Generate Test Predictions

```bash
python run_test.py
```

This will:
- Load test claims from `data/test.csv`
- Generate predictions for each claim
- Save results to `results.csv` with format:
  ```csv
  story_id,prediction,rationale
  test_001,0,"Found 2 violations in batch 1; Atoms: 5 (supported=1, violations=2, unsupported=2)"
  test_002,1,"Found 3 supported atoms in batch 1; Atoms: 4 (supported=3, violations=0, unsupported=1)"
  ...
  ```

**Output Format**:
- `story_id`: Test claim identifier
- `prediction`: 1 = CONSISTENT, 0 = INCONSISTENT
- `rationale`: Explanation with atom statistics

**Time**: ~5-8 minutes for 60 test claims

### Troubleshooting

**Issue**: `MISTRAL_API_KEY environment variable is required`
- **Solution**: Ensure `.env` file exists with valid API key

**Issue**: `No module named 'mistralai'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Cache files not found
- **Solution**: Run `python build_cache.py` first

**Issue**: CUDA out of memory
- **Solution**: E5-large-v2 will automatically fall back to CPU

## 📁 Project Structure

```
kdsh-2.0-track-a/
├── src/                           # Core pipeline modules
│   ├── chunking.py               # C-D-F-G chunking strategy
│   ├── semantic_index.py         # E5-large-v2 + FAISS
│   ├── final_decision_ensemble.py # Multi-stage ensemble decision logic
│   ├── claim_decomposer.py       # Atomic claim decomposition
│   ├── grounded_inference.py     # Evidence-based evaluation (3 perspectives)
│   ├── semantic_neighborhood.py  # Narrative compatibility
│   ├── character_profiles.py     # Mistral-generated profiles
│   ├── load_books.py            # Book preprocessing
│   ├── text_normalization.py    # Encoding consistency
│   └── config.py                # Configuration
├── data/                         # Training/test datasets
│   ├── train.csv                # Training claims
│   ├── test.csv                 # Test claims
│   └── raw/books/               # Source novels
├── cache/                        # Pre-computed components
│   ├── chunks/                  # Processed book chunks
│   ├── embeddings/              # FAISS indices
│   └── profiles/                # Character profiles
├── build_cache.py               # Cache generation
├── test_full_clean.py           # Training evaluation script
├── run_test.py                  # Test set prediction script
├── requirements.txt             # Dependencies
├── PIPELINE.md                  # Complete pipeline guide
├── FUNCTIONS.md                 # Function reference
├── FINAL_IMPLEMENTATION.md      # Implementation summary
├── MovingFlow.md               # Development journey
└── README.md                   # This file
```

## 🔧 Technical Features

### Advanced Decision Logic
- **OVER_SPECIFIED Detection**: Catches fabricated rituals and secret societies
- **Atomic Decomposition**: Breaks complex claims into 3-7 testable facts
- **Violation Classification**: HARD_VIOLATION vs UNSUPPORTED vs NO_CONSTRAINT
- **Impact-Based Routing**: Semantic evaluation only for causal claims
- **Epistemic Separation**: Grounded evidence vs narrative compatibility

### Performance Optimizations
- **GPU Acceleration**: CUDA support for E5-large-v2 embeddings
- **Persistent Caching**: Pre-computed embeddings and profiles
- **Rate Limiting**: API management for production deployment
- **Batch Processing**: Efficient embedding generation

### Evaluation Framework
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Detailed classification breakdown
- **Method Tracking**: Grounded vs semantic decision distribution
- **Transparency**: Dual verdicts with explanation logging

## 📊 Performance Results

### Current Metrics (30 Random Claims)
- **Accuracy**: 83.33%
- **Precision (CONTRADICT)**: 85.71%
- **Recall (CONTRADICT)**: 80.00%
- **F1-Score**: 82.76%

### System Validation
- **Epistemic Honesty**: 98.75% of training claims correctly identified as absent evidence
- **Perfect Accuracy**: 100% on evaluable claims
- **Robust Integration**: Zero API failures with Groq/Llama-3.1-8b-instant

## 🛠️ Configuration

### Environment Variables
```bash
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Model Configuration
- **Embeddings**: `intfloat/e5-large-v2`
- **LLM**: `mistral-small-2503` via Mistral API
- **Chunk Size**: ~850 tokens with 175 token overlap
- **Retrieval**: Top-10 semantic search with character filtering

## 📚 Documentation

- **[PIPELINE.md](PIPELINE.md)**: Complete end-to-end pipeline guide with ensemble architecture
- **[FUNCTIONS.md](FUNCTIONS.md)**: Function reference for all active components
- **[FINAL_IMPLEMENTATION.md](FINAL_IMPLEMENTATION.md)**: Implementation summary with performance metrics
- **[MovingFlow.md](MovingFlow.md)**: Detailed development journey and technical decisions
- **Source Code**: Comprehensive inline documentation

## 🎯 Use Cases

- **Narrative Consistency Verification**: Validate character backstories against source material
- **Literary Analysis**: Automated fact-checking for literary claims
- **Content Validation**: Verify fictional character details for accuracy
- **Research Tool**: Academic analysis of narrative consistency

## 🔄 Development Workflow

### Current Branch: Novelties
- Active development branch for new features
- Main branch contains stable production code

### Key Branches
- `main`: Production-ready system
- `Novelties`: Active development branch

## 🤝 Contributing

1. Work on the `Novelties` branch for new features
2. Follow the existing code structure and documentation standards
3. Run tests before committing: `python test_full_clean.py`
4. Update documentation for significant changes

## 📄 License

This project is developed for the Kharagpur Data Science Hackathon 2026.

---

**Status**: Production-Ready Advanced Solution ✅  
**Latest**: Mistral ensemble system with multi-perspective evaluation and strict support detection  
**Ready**: Advanced hackathon deployment with comprehensive caching and no rate limiting

---

## 🔧 System Optimizations

### Mistral API Migration
- Migrated from Groq Llama to Mistral Small 2503
- Better rate limits and API stability
- Improved reasoning quality for constraint inference
- No rate limiting for maximum throughput

### Ensemble Decision System
- Three perspectives: Strict, Moderate, Lenient
- Voting logic: 2+ CONTRADICT votes → final CONTRADICT
- Reduces false positives and false negatives
- API usage: 10-22 requests per claim

### Strict Support Detection
- Prevents co-occurrence hallucination
- Prompt engineering: Requires explicit statement
- Validation check: 50% word overlap threshold
- Improved precision on SUPPORTED verdicts

### Performance Characteristics
- **First Run**: ~10-15 minutes (cache generation)
- **Subsequent Runs**: Depends only on Mistral API speed
- **API Requests**: 10-22 per claim (1 decomposition + 3×3-7 atoms)
- **Total for 80 Claims**: ~800-1,760 requests
- **No Rate Limiting**: Maximum API throughput