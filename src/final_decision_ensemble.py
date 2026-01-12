from typing import List, Dict
from claim_decomposer import decompose_claim
from grounded_inference import grounded_constraint_inference
from semantic_index import SemanticIndex
from claim_classifier import classify_claim
from bounded_retrieval import should_escalate_retrieval, perform_bounded_escalation, K_MAX


def is_high_stakes_event(atom: str) -> bool:
    """Detect high-stakes events that should trigger HARD_VIOLATION if unsupported."""
    atom_lower = atom.lower()
    
    high_stakes_indicators = [
        # Major life events
        'arrested', 're-arrested', 'executed', 'died', 'killed', 'murdered', 'assassinated',
        'imprisoned', 'escaped', 'freed', 'released', 'exiled', 'banished',
        # Major meetings/interactions
        'met', 'encountered', 'confronted', 'visited', 'spoke with',
        # Organizations/conspiracies
        'joined', 'founded', 'established', 'member of', 'leader of', 'agent of',
        'secret society', 'conspiracy', 'underground', 'revolutionary',
        # Major discoveries
        'found', 'discovered', 'uncovered', 'revealed', 'exposed',
        'bribery', 'ledger', 'document', 'evidence', 'proof',
        # Family members (named)
        'sister', 'brother', 'son', 'daughter', 'wife', 'husband',
        # Aliases
        'alias', 'false name', 'disguise', 'posed as'
    ]
    
    return any(indicator in atom_lower for indicator in high_stakes_indicators)


def is_trivial_atom(atom: str) -> bool:
    """Detect trivial atoms that don't need ensemble evaluation."""
    atom_lower = atom.lower()
    trivial_patterns = [
        'had a', 'was a', 'were', 'family', 'childhood', 'boyhood',
        'mother', 'father', 'parent', 'sibling', 'gathering', 'celebration'
    ]
    return any(pattern in atom_lower for pattern in trivial_patterns)


def evaluate_with_ensemble(atom: str, evidence_chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """Evaluate atom with tuned threshold (0.45) and confidence calibration."""
    try:
        # Get verdicts from all 3 perspectives
        lenient_result = grounded_constraint_inference({'claim_text': atom}, evidence_chunks, perspective="lenient")
        moderate_result = grounded_constraint_inference({'claim_text': atom}, evidence_chunks, perspective="moderate")
        strict_result = grounded_constraint_inference({'claim_text': atom}, evidence_chunks, perspective="strict")
        
        lenient_verdict = lenient_result['verdict']
        moderate_verdict = moderate_result['verdict']
        strict_verdict = strict_result['verdict']
        
        # Priority 1: If ANY model says SUPPORTED, accept it
        if lenient_verdict == "SUPPORTED" or moderate_verdict == "SUPPORTED" or strict_verdict == "SUPPORTED":
            reason = lenient_result['reason'] if lenient_verdict == "SUPPORTED" else (moderate_result['reason'] if moderate_verdict == "SUPPORTED" else strict_result['reason'])
            return {"verdict": "SUPPORTED", "reason": reason}
        
        # D. Contradiction Threshold Tuning: Lower threshold from 0.5 to 0.45
        # Count HARD_VIOLATION votes
        violation_votes = sum([1 for v in [lenient_verdict, moderate_verdict, strict_verdict] if v == "HARD_VIOLATION"])
        
        # If 2+ models say HARD_VIOLATION (threshold: 2/3 = 0.67 > 0.45), flag as violation
        if violation_votes >= 2:
            return {"verdict": "HARD_VIOLATION", "reason": moderate_result['reason']}
        
        # E. Post-Processing: Confidence Calibration
        # If high-stakes event and UNSUPPORTED by all 3 → treat as HARD_VIOLATION
        if is_high_stakes_event(atom):
            unsupported_votes = sum([1 for v in [lenient_verdict, moderate_verdict, strict_verdict] if v == "UNSUPPORTED"])
            if unsupported_votes >= 2:  # Majority says UNSUPPORTED for high-stakes event
                return {"verdict": "HARD_VIOLATION", "reason": f"High-stakes event unsupported: {moderate_result['reason']}"}
        
        # Default: UNSUPPORTED (not enough evidence either way)
        return {"verdict": "UNSUPPORTED", "reason": lenient_result['reason']}
        
    except Exception:
        return {"verdict": "UNSUPPORTED", "reason": "API error"}


def aggregate_final_decision(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], semantic_index: SemanticIndex = None) -> Dict[str, str]:
    """Dual 3-agent system: one set finds contradictions, another finds consistency."""
    
    claim_text = claim.get('claim_text', '')
    
    # Decompose claim into atoms (reduced to 5 for efficiency)
    atoms = decompose_claim(claim_text)
    atoms = atoms[:5]  # Reduced from 7 to 5 atoms
    
    # Classify claim
    claim_classification = classify_claim(claim_text, atoms)
    
    # Progressive evaluation in batches: 1, 3, 6 (10 chunks total)
    batch_sizes = [1, 3, 6]
    start_idx = 0
    
    contradict_batch = None
    consistent_batch = None
    contradict_result = None
    consistent_result = None
    
    for batch_num in range(3):
        batch_size = batch_sizes[batch_num]
        end_idx = start_idx + batch_size
        batch_chunks = evidence_chunks[start_idx:end_idx]
        
        if len(batch_chunks) == 0:
            break
        
        violations = []
        supports = []
        atom_details = []
        
        # Evaluate each atom with current batch
        for atom in atoms:
            is_high_stakes = is_high_stakes_event(atom)
            is_trivial = is_trivial_atom(atom)
            
            if is_trivial and not is_high_stakes:
                result = grounded_constraint_inference({'claim_text': atom}, batch_chunks)
                verdict = result['verdict']
                reason = result['reason']
            else:
                result = evaluate_with_ensemble(atom, batch_chunks)
                verdict = result['verdict']
                reason = result['reason']
            
            atom_details.append({
                'atom': atom,
                'verdict': verdict,
                'reason': reason,
                'is_violation': verdict == "HARD_VIOLATION" or (is_high_stakes and verdict != "SUPPORTED"),
                'is_support': verdict == "SUPPORTED"
            })
            
            if verdict == "HARD_VIOLATION":
                violations.append({'atom': atom, 'reason': reason})
            elif is_high_stakes and verdict != "SUPPORTED":
                violations.append({'atom': atom, 'reason': reason})
            
            if verdict == "SUPPORTED":
                supports.append({'atom': atom, 'reason': reason})
        
        # Track first batch with contradictions (1+ violations)
        if len(violations) > 0 and contradict_batch is None:
            contradict_batch = batch_num + 1
            contradict_result = {
                "final_decision": "contradict",
                "explanation": f"Found {len(violations)} violations in batch {batch_num + 1}",
                "grounded_verdict": "CONTRADICT",
                "semantic_verdict": None,
                "method": "ENSEMBLE_CONTRADICT",
                "atoms_evaluated": len(atoms),
                "violations": len(violations),
                "atom_details": atom_details,
                "violation_atoms": violations,
                "expected_evidence": claim_classification['expected_evidence'],
                "batch_evaluated": batch_num + 1,
                "chunks_examined": end_idx
            }
        
        # Track first batch with consistency
        if len(supports) > 0 and consistent_batch is None:
            consistent_batch = batch_num + 1
            consistent_result = {
                "final_decision": "consistent",
                "explanation": f"Found {len(supports)} supported atoms in batch {batch_num + 1}",
                "grounded_verdict": "CONSISTENT",
                "semantic_verdict": None,
                "method": "ENSEMBLE_CONSISTENT",
                "atoms_evaluated": len(atoms),
                "violations": 0,
                "atom_details": atom_details,
                "violation_atoms": [],
                "support_atoms": supports,
                "expected_evidence": claim_classification['expected_evidence'],
                "batch_evaluated": batch_num + 1,
                "chunks_examined": end_idx
            }
        
        start_idx = end_idx
    
    # Decision logic with support-bias: prioritize consistency when evidence is mixed
    if contradict_batch is not None and consistent_batch is not None:
        # Both found evidence - apply support-bias
        # If support found in same or earlier batch, prioritize consistency
        if consistent_batch <= contradict_batch:
            return consistent_result
        # If contradiction found significantly earlier (2+ batches), accept it
        elif contradict_batch < consistent_batch - 1:
            return contradict_result
        else:
            # Close call - favor consistency (benefit of doubt)
            return consistent_result
    elif contradict_batch is not None:
        # Only contradiction found - but require 2+ violations for high confidence
        if contradict_result['violations'] >= 2:
            return contradict_result
        else:
            # Single violation - give benefit of doubt
            return {
                "final_decision": "consistent",
                "explanation": f"Only 1 violation found (benefit of doubt given)",
                "grounded_verdict": "CONSISTENT",
                "semantic_verdict": None,
                "method": "ENSEMBLE_LENIENT",
                "atoms_evaluated": len(atoms),
                "violations": 1,
                "atom_details": contradict_result['atom_details'],
                "violation_atoms": contradict_result['violation_atoms'],
                "expected_evidence": claim_classification['expected_evidence'],
                "batch_evaluated": contradict_batch,
                "chunks_examined": contradict_result['chunks_examined']
            }
    elif consistent_batch is not None:
        # Only consistency found
        return consistent_result
    else:
        # No evidence found - default to consistent (benefit of doubt)
        return {
            "final_decision": "consistent",
            "explanation": "No violations or support found in 10 chunks (benefit of doubt)",
            "grounded_verdict": "CONSISTENT",
            "semantic_verdict": None,
            "method": "ENSEMBLE_DEFAULT",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "atom_details": [],
            "violation_atoms": [],
            "expected_evidence": claim_classification['expected_evidence'],
            "batch_evaluated": 3,
            "chunks_examined": 10
        }
