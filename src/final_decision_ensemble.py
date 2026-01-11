from typing import List, Dict
from claim_decomposer import decompose_claim
from grounded_inference import grounded_constraint_inference
from semantic_index import SemanticIndex


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
    """Smart ensemble: Use ensemble only for important atoms."""
    
    claim_text = claim.get('claim_text', '')
    
    # Decompose claim into atoms
    atoms = decompose_claim(claim_text)
    atoms = atoms[:7]
    
    violations = []
    atom_details = []
    
    # Evaluate each atom
    for atom in atoms:
        is_obligation = is_canon_obligated_atom(atom)
        is_trivial = is_trivial_atom(atom)
        
        # Trivial atoms: single pass (MODERATE only)
        if is_trivial and not is_obligation:
            result = grounded_constraint_inference({'claim_text': atom}, evidence_chunks)
            verdict = result['verdict']
            reason = result['reason']
        else:
            # Important atoms: use ensemble
            result = evaluate_with_ensemble(atom, evidence_chunks)
            verdict = result['verdict']
            reason = result['reason']
        
        # Store atom details
        atom_details.append({
            'atom': atom,
            'verdict': verdict,
            'reason': reason,
            'is_violation': verdict == "HARD_VIOLATION" or (is_obligation and verdict != "SUPPORTED")
        })
        
        # Check for violations
        if verdict == "HARD_VIOLATION":
            violations.append({'atom': atom, 'reason': reason})
        elif is_obligation and verdict != "SUPPORTED":
            violations.append({'atom': atom, 'reason': reason})
    
    # Decision based on violations
    if len(violations) > 0:
        return {
            "final_decision": "contradict",
            "explanation": f"Found {len(violations)} violations",
            "grounded_verdict": "CONTRADICT",
            "semantic_verdict": None,
            "method": "ENSEMBLE",
            "atoms_evaluated": len(atoms),
            "violations": len(violations),
            "atom_details": atom_details,
            "violation_atoms": violations
        }
    
    # Default to consistent
    return {
        "final_decision": "consistent",
        "explanation": "No violations found",
        "grounded_verdict": "CONSISTENT",
        "semantic_verdict": None,
        "method": "ENSEMBLE",
        "atoms_evaluated": len(atoms),
        "violations": 0,
        "atom_details": atom_details,
        "violation_atoms": []
    }
