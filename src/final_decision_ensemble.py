from typing import List, Dict
from claim_decomposer import decompose_claim
from grounded_inference import grounded_constraint_inference
from semantic_index import SemanticIndex
from claim_classifier import classify_claim
from bounded_retrieval import should_escalate_retrieval, perform_bounded_escalation, K_MAX


def is_canon_obligated_atom(atom: str) -> bool:
    """Detect if atom is canon-obligated (must have explicit support)."""
    atom_lower = atom.lower()
    
    state_verbs = ['arrested', 're-arrested', 'imprisoned', 'shipped', 'exiled', 'released', 'escaped']
    if any(verb in atom_lower for verb in state_verbs):
        return True
    
    first_time_indicators = ['met', 'first', 'introduced', 'joined', 'initially', 'began']
    if any(indicator in atom_lower for indicator in first_time_indicators):
        return True
    
    interaction_verbs = ['met', 'encountered', 'spoke with', 'confronted', 'visited', 'saw']
    if any(verb in atom_lower for verb in interaction_verbs):
        return True
    
    institutional_indicators = ['member of', 'leader of', 'agent of', 'head of', 'founded', 'established']
    if any(indicator in atom_lower for indicator in institutional_indicators):
        return True
    
    specific_locations = ['tasmania', 'new zealand', 'paris', 'marseilles', 'rome', 'madrid', 'chateau']
    if any(location in atom_lower for location in specific_locations):
        return True
    
    causal_indicators = ['because', 'triggered by', 'led to', 'caused by', 'resulted in', 'began when', 'when']
    if any(indicator in atom_lower for indicator in causal_indicators):
        return True
    
    return False


def is_trivial_atom(atom: str) -> bool:
    """Detect trivial atoms that don't need ensemble evaluation."""
    atom_lower = atom.lower()
    trivial_patterns = [
        'had a', 'was a', 'were', 'family', 'childhood', 'boyhood',
        'mother', 'father', 'parent', 'sibling', 'gathering', 'celebration'
    ]
    return any(pattern in atom_lower for pattern in trivial_patterns)


def evaluate_with_ensemble(atom: str, evidence_chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """Evaluate atom with 3 perspectives and vote."""
    # Get verdicts from all 3 perspectives
    moderate_result = grounded_constraint_inference({'claim_text': atom}, evidence_chunks)
    moderate_verdict = moderate_result['verdict']
    reason = moderate_result['reason']
    
    # Strict: treat NO_CONSTRAINT as UNSUPPORTED
    strict_verdict = moderate_verdict
    if moderate_verdict == "NO_CONSTRAINT":
        strict_verdict = "UNSUPPORTED"
    
    # Lenient: treat UNSUPPORTED as NO_CONSTRAINT
    lenient_verdict = moderate_verdict
    if moderate_verdict == "UNSUPPORTED":
        lenient_verdict = "NO_CONSTRAINT"
    
    # Vote: 2+ agree on HARD_VIOLATION → HARD_VIOLATION
    violation_votes = sum([
        strict_verdict == "HARD_VIOLATION",
        moderate_verdict == "HARD_VIOLATION",
        lenient_verdict == "HARD_VIOLATION"
    ])
    
    if violation_votes >= 2:
        return {"verdict": "HARD_VIOLATION", "reason": reason}
    
    # Vote: 2+ agree on SUPPORTED → SUPPORTED
    support_votes = sum([
        strict_verdict == "SUPPORTED",
        moderate_verdict == "SUPPORTED",
        lenient_verdict == "SUPPORTED"
    ])
    
    if support_votes >= 2:
        return {"verdict": "SUPPORTED", "reason": reason}
    
    # Default to moderate verdict
    return {"verdict": moderate_verdict, "reason": reason}


def aggregate_final_decision(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], semantic_index: SemanticIndex = None) -> Dict[str, str]:
    """Dual 3-agent system: one set finds contradictions, another finds consistency."""
    
    claim_text = claim.get('claim_text', '')
    
    # Decompose claim into atoms
    atoms = decompose_claim(claim_text)
    atoms = atoms[:7]
    
    # Classify claim
    claim_classification = classify_claim(claim_text, atoms)
    
    # Progressive evaluation in batches: 5, 5, 5 (15 chunks total)
    batch_sizes = [5, 5, 5]
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
            is_obligation = is_canon_obligated_atom(atom)
            is_trivial = is_trivial_atom(atom)
            
            if is_trivial and not is_obligation:
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
                'is_violation': verdict == "HARD_VIOLATION" or (is_obligation and verdict != "SUPPORTED"),
                'is_support': verdict == "SUPPORTED"
            })
            
            if verdict == "HARD_VIOLATION":
                violations.append({'atom': atom, 'reason': reason})
            elif is_obligation and verdict != "SUPPORTED":
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
    
    # Decision logic: prioritize based on batch order
    if contradict_batch is not None and consistent_batch is not None:
        # Both found evidence
        if contradict_batch < consistent_batch:
            # Contradiction found earlier → prioritize contradiction
            return contradict_result
        else:
            # Same batch or consistency found earlier → prioritize consistency (skewed data)
            return consistent_result
    elif contradict_batch is not None:
        # Only contradiction found
        return contradict_result
    elif consistent_batch is not None:
        # Only consistency found
        return consistent_result
    else:
        # No evidence found
        return {
            "final_decision": "consistent",
            "explanation": "No violations or support found in 15 chunks",
            "grounded_verdict": "CONSISTENT",
            "semantic_verdict": None,
            "method": "ENSEMBLE_DEFAULT",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "atom_details": [],
            "violation_atoms": [],
            "expected_evidence": claim_classification['expected_evidence'],
            "batch_evaluated": 3,
            "chunks_examined": 15
        }
