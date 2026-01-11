# Failed Experiment: Violation Strength & Scoring System

## Date
January 12, 2026

## Experiment Overview
Attempted to improve the dual-agent 3-3-4 batch system by introducing:
1. Violation strength levels (STRONG, WEAK, NONE)
2. Canon silence rule for high-salience events
3. Weighted scoring system
4. Contradiction gate logic

## Baseline Performance (Before Changes)
- **Accuracy**: 63.75% (51/80 correct)
- **Precision**: Balanced
- **Recall**: 48%
- **F1-Score**: ~50%
- **Architecture**: Dual-agent 3-3-4 batch system

## Changes Made

### 1. Violation Strength Classification
```python
STRONG_VIOLATION = +3  # Explicit competing facts
WEAK_VIOLATION = +1    # Missing details, partial mismatch
CANON_SILENCE = +3     # High-salience events missing
SUPPORT = -2           # Explicit support found
```

### 2. Canon Silence Rule
Flagged claims with high-salience events (arrests, first meetings, secret societies) as violations if not found in canon.

### 3. Scoring System
```python
score = (
    3 * strong_violations +
    1 * weak_violations +
    3 * canon_silence -
    2 * supports
)

if score >= 2: CONTRADICT
elif score <= -2: CONSISTENT
else: CONSISTENT (tie-breaker)
```

### 4. Simplified Decision Logic
Removed batch-based early termination, evaluated all atoms across all 10 chunks, then applied scoring.

## Results (After Changes)
- **Accuracy**: WORSENED (dropped significantly)
- **F1-Score**: 11% (catastrophic drop from ~50%)
- **Precision**: Collapsed
- **Recall**: Collapsed

## Root Causes of Failure

### 1. Canon Silence Over-Triggering
- Canon silence detector fired on biographical elaborations
- Political affiliations (Girondins) flagged as violations
- Birthplace details flagged as violations
- **Problem**: Treated any unsupported atom as "canon silence"
- **Impact**: Massive false positive rate

### 2. Obligation Classification Leakage
- Obligation detection was too broad
- Generic interactions ("met", "saw") flagged as obligations
- Locations (Tasmania, Paris) flagged as obligations
- Causal indicators ("because", "when") flagged as obligations
- **Problem**: Rare obligations became common
- **Impact**: System required explicit support for texture details

### 3. Blind System (0 Supports, 0 Violations)
- Many claims produced score = 0
- System defaulted to CONSISTENT without evidence
- **Problem**: Support detector failing to match obvious evidence
- **Impact**: Low recall, system couldn't "see" evidence

### 4. Asymmetric Logic Amplified
- Scoring system amplified existing asymmetries
- False positives accumulated faster than before
- Weak violations outweighed supports incorrectly
- **Problem**: Weights were arbitrary, not calibrated
- **Impact**: System became more unstable, not less

### 5. Removed Batch-Based Early Termination
- Original system stopped at first contradiction/support
- New system evaluated all atoms, accumulated noise
- **Problem**: Lost the benefit of progressive evaluation
- **Impact**: More API calls, more noise, worse performance

## Key Lessons Learned

### 1. Complexity Without Foundation
Adding weighted scoring on top of broken detectors amplified errors rather than fixing them.

### 2. Canon Silence Must Be Rare
Canon silence should apply to <5% of claims (arrests, deaths, executions), not 30-40% (affiliations, locations, interactions).

### 3. Obligation Detection Must Be Narrow
Only state changes (arrested, executed) and explicit first meetings with named characters should be obligations.

### 4. Support Detection Is Critical
If the system produces 0 supports for claims that should have support, no amount of violation logic will help.

### 5. Batch-Based Early Termination Was Valuable
Progressive evaluation with early stopping prevented noise accumulation.

### 6. Accuracy "Drop" Was Misunderstood
The initial accuracy "drop" from 65% to 63.75% was actually progress (better recall). The real drop came from broken detectors, not from the concept itself.

## What Should Have Been Done Instead

### 1. Fix Support Detection First
Before adding violation logic, ensure the system can find obvious supports.

### 2. Validate Detectors Independently
Test canon silence and obligation detectors on sample claims before integrating.

### 3. Keep Batch-Based Logic
Progressive evaluation with early termination was working - don't remove it.

### 4. Calibrate Weights Empirically
If using scoring, calibrate weights on validation set, don't guess.

### 5. One Change at a Time
Test each change independently before combining.

## Conclusion

This experiment demonstrated that:
- **Adding complexity without fixing foundations makes things worse**
- **Detector quality matters more than decision logic**
- **The dual-agent 3-3-4 batch system was already near-optimal for the given detectors**
- **Improving from 63.75% requires better evidence detection, not better decision rules**

The system has been reverted to the baseline dual-agent architecture (63.75% accuracy, 48% recall, ~50% F1).

## Recommendation

To improve beyond 63.75%, focus on:
1. **Better support detection**: Why is the system producing 0 supports?
2. **Better LLM prompts**: Improve grounded_inference.py prompts
3. **Better chunking**: Ensure relevant evidence is in retrieved chunks
4. **Better embeddings**: Consider fine-tuning E5 or using better retrieval

Do NOT:
- Add more violation classification logic
- Add more scoring systems
- Add more rules without validating detectors first
- Remove batch-based early termination

---

**Status**: Reverted to commit 53a209a (baseline dual-agent system)  
**Performance**: 63.75% accuracy, 48% recall, ~50% F1  
**Next Steps**: Focus on evidence detection, not decision logic
