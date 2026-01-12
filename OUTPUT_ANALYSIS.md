# Output Pattern Analysis: Path to Improvement

## Performance Trajectory
- **Claims 1-20**: 75% accuracy, 66.67% F1 (STRONG START)
- **Claims 21-40**: 72.5% accuracy, 64.52% F1 (slight decline)
- **Claims 41-60**: 65% accuracy, 51.16% F1 (significant drop)
- **Final (80 claims)**: 63.75% accuracy, ~50% F1

## Critical Pattern #1: ENSEMBLE_DEFAULT Dominates

### The Problem
**56 out of 80 claims** (70%) result in:
```
Method: ENSEMBLE_DEFAULT
Explanation: No violations or support found in 10 chunks
```

This means the system is **blind** - it's not finding evidence for most claims.

### Why This Matters
- When system finds no evidence, it defaults to CONSISTENT
- This works for true CONSISTENT claims (lucky guess)
- This fails catastrophically for true CONTRADICT claims (missed violations)

### Examples of Missed Contradictions (False Negatives)
1. **Claim 19** (ID 12): "declined to take part in charting maiden voyage"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Major event (declining voyage) should be findable

2. **Claim 28** (ID 89): "Arguing for procedural justice at Louis XVI trial"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Historical event should have evidence

3. **Claim 32** (ID 116): "backed constitutionalist Prince Pedro, branded traitor"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Major political event missing

4. **Claim 42** (ID 113): "Born in Parma to theological family"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Birthplace contradiction not detected

5. **Claim 44** (ID 71): "let troops burn empty village as decoy"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Military tactic should be verifiable

6. **Claim 46** (ID 117): "hid lifetime research manuscripts in Madrid monastery"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Major plot point missing

7. **Claim 49** (ID 17): "father died early; mother remarried French officer"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Family background contradiction

8. **Claim 52** (ID 122): "watched young prosecutor Villefort at Vienna-congress"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Character meeting contradiction

9. **Claim 56** (ID 98): "Hidden Waterloo-era diplomatic letters"
   - True: CONTRADICT
   - Predicted: CONSISTENT (ENSEMBLE_DEFAULT)
   - **Issue**: Plot element missing

**Pattern**: System is missing 9+ contradictions because it finds NO evidence at all.

## Critical Pattern #2: False Positives from Single Violations

### The Problem
System triggers CONTRADICT with just 1 violation, even for texture details.

### Examples of False Positives
1. **Claim 9** (ID 68): "flame-shaped birth-mark on left shoulder"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Texture detail flagged as violation

2. **Claim 14** (ID 88): "joined Girondins, drafted manifestos"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Political affiliation flagged as violation

3. **Claim 15** (ID 67): "Born on New Zealand's North-island east coast"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Geographic detail flagged as violation

4. **Claim 30** (ID 99): "Born into Parisian legal family"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Background detail flagged as violation

5. **Claim 31** (ID 123): "studied Napoleon's secret-police tactics"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Study topic flagged as violation

6. **Claim 35** (ID 128): "invisible-ink formula from temple-mural restoration"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Origin story flagged as violation

7. **Claim 47** (ID 6): "met young Captain Grant in Marseille waterfront bar"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Meeting location flagged as violation

8. **Claim 57** (ID 25): "met ex-General von Waldeck on eve of sailing"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Meeting detail flagged as violation

9. **Claim 58** (ID 120): "invented salt-blood ink revealed under heat"
   - True: CONSISTENT
   - Predicted: CONTRADICT (1 violation in batch 1)
   - **Issue**: Invention detail flagged as violation

10. **Claim 60** (ID 57): "father vanished in local uprising at age seven"
    - True: CONSISTENT
    - Predicted: CONTRADICT (2 violations in batch 1)
    - **Issue**: Background trauma flagged as violation

**Pattern**: System is too aggressive - single weak violations trigger CONTRADICT.

## Critical Pattern #3: Rare Correct Support Detection

### The Problem
Only **3 claims** found support (ENSEMBLE_CONSISTENT):
- Claim 4 (ID 109) - FALSE POSITIVE (found support for CONTRADICT claim)
- Claim 34 (ID 108) - TRUE POSITIVE
- Claim 40 (ID 39) - FALSE POSITIVE (found support for CONTRADICT claim)
- Claim 50 (ID 20) - TRUE POSITIVE

**Success rate**: 2/4 = 50% when support is found

### Why This Matters
Support detection is broken - it rarely fires, and when it does, it's wrong 50% of the time.

## Root Causes

### 1. Retrieval Failure (70% of claims)
**Evidence**: 56/80 claims produce "No violations or support found"

**Possible causes**:
- Semantic search not finding relevant chunks
- Character filtering too aggressive
- Chunks don't contain the relevant evidence
- Query embedding not matching passage embeddings

**Impact**: System is blind to most evidence

### 2. Over-Aggressive Violation Detection (12.5% false positive rate)
**Evidence**: 10/80 claims are false positives from single violations

**Possible causes**:
- LLM prompt too strict (treats texture as violation)
- No distinction between "not mentioned" and "contradicted"
- Ensemble voting too sensitive (1 violation triggers CONTRADICT)

**Impact**: Precision drops from 62.5% to 52.38%

### 3. Broken Support Detection (rare and unreliable)
**Evidence**: Only 4 claims found support, 50% were wrong

**Possible causes**:
- LLM prompt too conservative (requires exact match)
- 50% word overlap validation too strict
- Support requires explicit statement, paraphrase not accepted

**Impact**: Recall drops from 71.43% to 50%

## Recommended Fixes (Priority Order)

### Fix #1: Improve Retrieval (HIGHEST PRIORITY)
**Problem**: 70% of claims find no evidence at all

**Solutions**:
1. **Increase retrieved chunks**: Try 15-20 chunks instead of 10
2. **Remove character filtering**: May be too aggressive
3. **Better query construction**: Include character name + key claim terms
4. **Hybrid retrieval**: Add keyword search alongside semantic search
5. **Check chunk quality**: Verify chunks contain relevant narrative

**Expected impact**: +10-15% accuracy (most claims will find evidence)

### Fix #2: Relax Support Detection (HIGH PRIORITY)
**Problem**: Support detection too conservative, only fires 5% of the time

**Solutions**:
1. **Remove 50% word overlap validation**: Too strict
2. **Accept paraphrase**: "arrested" = "taken into custody"
3. **Accept partial support**: If 2/3 atoms supported, claim is supported
4. **Better LLM prompt**: Emphasize finding support, not just violations

**Expected impact**: +5-8% accuracy (better recall)

### Fix #3: Add Violation Strength Threshold (MEDIUM PRIORITY)
**Problem**: Single weak violations trigger CONTRADICT

**Solutions**:
1. **Require 2+ violations for CONTRADICT**: Unless violation is STRONG
2. **Classify violation strength**: STRONG (competing facts) vs WEAK (missing details)
3. **Support can override weak violations**: 1 support + 1 weak violation = CONSISTENT

**Expected impact**: +3-5% accuracy (better precision)

### Fix #4: Better Atom Decomposition (LOW PRIORITY)
**Problem**: Some atoms may be too granular or too broad

**Solutions**:
1. **Validate atom quality**: Check if atoms are testable
2. **Merge related atoms**: Avoid splitting single facts
3. **Filter trivial atoms**: Remove texture details from evaluation

**Expected impact**: +1-2% accuracy (cleaner evaluation)

## Immediate Action Plan

### Step 1: Increase Retrieved Chunks (Quick Win)
```python
# In test_full_clean.py and run_test.py
evidence_chunks = semantic_index.semantic_retrieve(claim, max_chunks=15)  # was 10
```

### Step 2: Remove Word Overlap Validation (Quick Win)
```python
# In grounded_inference.py, remove this block:
if "SUPPORTED" in verdict:
    # Remove the 50% word overlap check
    pass
```

### Step 3: Require 2+ Violations for CONTRADICT
```python
# In final_decision_ensemble.py
if len(violations) >= 2:  # was >= 1
    return CONTRADICT
```

### Step 4: Test and Measure
Run test_full_clean.py after each change and track:
- Accuracy
- Precision
- Recall
- F1-Score
- Method distribution (ENSEMBLE_DEFAULT should drop below 50%)

## Expected Results After Fixes

### Current Baseline
- Accuracy: 63.75%
- Precision: 52.38%
- Recall: 50%
- F1: 51.16%
- ENSEMBLE_DEFAULT: 70%

### After Fix #1 (More Chunks)
- Accuracy: 70-75%
- ENSEMBLE_DEFAULT: 50-60%
- More evidence found

### After Fix #2 (Relax Support)
- Accuracy: 75-78%
- Recall: 60-65%
- More supports detected

### After Fix #3 (Violation Threshold)
- Accuracy: 78-82%
- Precision: 65-70%
- Fewer false positives

## Key Insight

**The system is not broken - it's blind.**

70% of claims produce no evidence. This is a retrieval problem, not a decision logic problem.

Fix retrieval first, then tune decision logic.
