# KDSH 2.0 Track-A: Narrative Consistency Verification System - Final Model

A sophisticated system for verifying the consistency of character backstories against full novel narratives using multi-stage decision logic, semantic embeddings, and LLM-based reasoning.

## 🎯 Project Overview

**Goal**: Determine whether hypothetical character backstories are causally and logically consistent with full narratives of long-form novels (100k+ words).

**Approach**: Multi-stage verification system with 3-perspective ensemble, character-specific retrieval, and high-stakes calibration.

## 🏗️ System Architecture

### Core Pipeline
```
Raw Books → C-D-F-G Chunking → E5-Large-v2 Embeddings → FAISS Indexing → 
Character/Temporal Boosted Retrieval (15 chunks) → Atomic Decomposition (5 atoms) → 
3-Perspective Ensemble (strict/moderate/lenient) → High-Stakes Calibration → 
Threshold Tuning (2/3 votes) → Final Decision
```

### Key Components

- **📚 Advanced Chunking**: C-D-F-G strategy with sliding windows and character tagging
- **🔍 Boosted Retrieval**: Character-specific (2x) and temporal filtering (1.5x)
- **🧠 Plot-Critical Atoms**: 5 verifiable facts (names, dates, events, locations, causes)
- **⚖️ 3-Perspective Ensemble**: Strict/moderate/lenient with 2/3 voting threshold
- **🎯 High-Stakes Calibration**: Unsupported plot-critical events → HARD_VIOLATION
- **📊 Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 🚀 Quick Start (Reproducible)

### Prerequisites
```bash
pip install -r requirements.txt
```

**Required packages**:
- `mistralai==1.2.4`
- `sentence-transformers==3.3.1`
- `faiss-cpu==1.9.0` (or `faiss-gpu`)
- `torch==2.5.1`
- `python-dotenv==1.0.1`
- `numpy==2.2.0`
- `pandas==2.2.3`

### Step 1: Setup API Key

Create `.env` file:
```bash
MISTRAL_API_KEY=your_actual_api_key
```

Get key at: https://console.mistral.ai/

### Step 2: Build Cache (One-time, ~10-15 min)

```bash
python build_cache.py
```

Generates:
- `cache/chunks/` - Book chunks
- `cache/embeddings/` - FAISS indices

### Step 3: Run Training Evaluation (~20-30 min)

```bash
python test_full_clean.py
```

Expected output:
```
=== Dual Agent System: 15 Chunks (1-3-6 batches) ===
Loaded 80 total claims
...
=== RESULTS ===
Correct: 54/80
Accuracy: 67.50%
Precision: 65.00%
Recall: 52.00%
F1-Score: 57.78%
```

### Step 4: Generate Test Predictions (~15-20 min)

```bash
python run_test.py
```

Output: `results.csv` with format:
```csv
story_id,prediction,rationale
test_001,0,"Found 2 violations..."
test_002,1,"Found 3 supported atoms..."
```

## 📊 Performance Results

### Final Metrics (80 Training Claims)
- **Accuracy**: 67.50%
- **Precision**: 65.00%
- **Recall**: 52.00%
- **F1-Score**: 57.78%

### API Usage
- **Per Claim**: ~30-45 calls (1 decomposition + 3 perspectives × 5 atoms × 3 batches)
- **80 Claims**: ~2,400-3,600 calls
- **Time**: ~20-30 minutes
- **Cost**: ~$1-2 (Mistral Small 2503)

## 🔧 Key Optimizations

1. **Character-Specific Retrieval Boost (2x)**: +1-2% accuracy
2. **Temporal Filtering (1.5x)**: +1-2% accuracy
3. **Contradiction Threshold Tuning (2/3 votes)**: +1-3% accuracy
4. **High-Stakes Calibration**: +1-2% accuracy
5. **Plot-Critical Atoms (5 focused facts)**: +3-5% accuracy

**Total Expected Improvement**: +7-14% over baseline

## 🛠️ Configuration

### Model Settings
- **Embeddings**: `intfloat/e5-large-v2` (local, GPU/CPU)
- **LLM**: `mistral-small-2503` (API)
- **Chunks**: 15 retrieved, ~850 tokens each
- **Atoms**: 5 plot-critical facts
- **Ensemble**: 3 perspectives (strict/moderate/lenient)
- **Timeout**: 30s per API call

### Environment Variables
```bash
MISTRAL_API_KEY=your_key_here
```

## 📁 Project Structure

```
kdsh-2.0/
├── src/
│   ├── semantic_index.py         # Retrieval with character/temporal boost
│   ├── final_decision_ensemble.py # 3-perspective ensemble
│   ├── claim_decomposer.py       # Plot-critical atom extraction
│   ├── grounded_inference.py     # Mistral API integration
│   ├── claim_classifier.py
│   ├── bounded_retrieval.py
│   ├── chunking.py
│   ├── load_books.py
│   └── config.py
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── raw/books/
├── cache/
│   ├── chunks/
│   └── embeddings/
├── build_cache.py
├── test_full_clean.py
├── run_test.py
├── requirements.txt
└── README.md
```

## 🔄 Reproducibility Checklist

- ✅ Install exact package versions from `requirements.txt`
- ✅ Set `MISTRAL_API_KEY` in `.env` file
- ✅ Run `build_cache.py` to generate embeddings
- ✅ Run `test_full_clean.py` for training evaluation
- ✅ Run `run_test.py` for test predictions
- ✅ Results saved to `results.csv`

## 📄 License

Kharagpur Data Science Hackathon 2026

---

**Status**: Final Production Model ✅  
**Branch**: `final_model`  
**Accuracy**: 67.50% (target: 75%+)  
**Last Updated**: 2025
