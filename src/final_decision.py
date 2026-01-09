from typing import List, Dict
import time
import re
from claim_decomposer import decompose_claim
from grounded_inference import grounded_constraint_inference
from semantic_neighborhood import semantic_neighborhood_evaluation
from semantic_index import SemanticIndex


def classify_claim_salience(claim_text: str) -> str:
    """
    Classify claim salience based on content type (more conservative).
    
    Returns:
        "HIGH" or "LOW"
    """
    claim_lower = claim_text.lower()
    
    # High salience indicators (more restrictive)
    high_salience_markers = [
        # Political allegiance/betrayal
        'betrayal', 'traitor', 'conspiracy', 'plot', 'revolutionary', 'ally', 'enemy',
        # Executions/arrests/exile
        'execution', 'executed', 'arrest', 'arrested', 'exile', 'exiled', 'imprisoned',
        # Major timeline events
        'revolution', 'war', 'battle', 'death', 'died', 'murder', 'killed',
        # Secret societies/conspiracies
        'secret society', 'conspiracy', 'conspirator', 'secret meeting',
        # Irreversible life events
        'married', 'divorced', 'born', 'founded', 'established', 'destroyed'
    ]
    
    # Count high salience markers
    salience_count = sum(1 for marker in high_salience_markers if marker in claim_lower)
    
    return "HIGH" if salience_count >= 1 else "LOW"


def classify_atom_weight(atom: str) -> float:
    """
    Assign weight to atom based on content importance.
    
    Returns:
        Weight value (0.5, 1.5, or 2.0)
    """
    atom_lower = atom.lower()
    
    # Core identity / timeline facts (weight 2.0)
    core_markers = [
        'born', 'died', 'age', 'family', 'father', 'mother', 'son', 'daughter',
        'married', 'childhood', 'youth', 'timeline', 'year', 'date'
    ]
    
    # Political allegiance / major actions (weight 1.5)
    political_markers = [
        'revolution', 'war', 'battle', 'political', 'ally', 'enemy', 'betrayal',
        'king', 'emperor', 'noble', 'minister', 'leader', 'founded', 'ruled'
    ]
    
    # Check for core identity markers
    if any(marker in atom_lower for marker in core_markers):
        return 2.0
    
    # Check for political markers
    if any(marker in atom_lower for marker in political_markers):
        return 1.5
    
    # Default: emotional, symbolic, stylistic details
    return 0.5


def is_strong_expected_atom(atom: str, weight: float) -> bool:
    """
    Check if atom is STRONG_EXPECTED based on weight and type.
    
    Returns:
        True if atom requires strong grounding
    """
    if weight < 1.5:
        return False
    
    atom_lower = atom.lower()
    strong_types = [
        # Political role/allegiance
        'political', 'ally', 'enemy', 'revolutionary', 'loyalist', 'minister', 'leader',
        # Secret society/conspiracy
        'secret', 'society', 'conspiracy', 'plot', 'cabal', 'clandestine',
        # Arrest/execution/exile
        'arrest', 'execution', 'exile', 'imprisoned', 'executed', 'banished',
        # Betrayal/exposure
        'betrayal', 'betrayed', 'exposed', 'revealed', 'traitor',
        # Major relationship change
        'married', 'divorced', 'alliance', 'partnership',
        # Public historical event
        'revolution', 'war', 'battle', 'treaty', 'founded', 'established'
    ]
    
    return any(marker in atom_lower for marker in strong_types)


def contains_covert_keywords(claim_text: str) -> bool:
    """
    Check if claim contains covert/secret keywords.
    
    Returns:
        True if covert keywords found
    """
    claim_lower = claim_text.lower()
    covert_keywords = [
        'clandestine', 'secret', 'covert', 'underground', 'hidden',
        'forged', 'tampered', 'society', 'plot', 'cabal'
    ]
    
    return any(keyword in claim_lower for keyword in covert_keywords)


def check_timeline_violation(claim_text: str, character: str) -> bool:
    """
    Check for timeline violations using rule-based logic.
    
    Returns:
        True if timeline violation detected
    """
    claim_lower = claim_text.lower()
    character_lower = character.lower()
    
    # Timeline bins for major characters (simplified)
    timeline_bins = {
        'edmund_dantes': {
            'PRE_REVOLUTION': ['youth', 'childhood', 'early life', 'before prison'],
            'IMPRISONMENT': ['prison', 'chateau d\'if', 'imprisoned', 'captive'],
            'POST_ESCAPE': ['count', 'monte cristo', 'revenge', 'after escape'],
            'EXILE': ['exile', 'banished', 'fled']
        },
        'fernand_mondego': {
            'PRE_REVOLUTION': ['youth', 'childhood', 'early life'],
            'REVOLUTIONARY': ['revolution', 'war', 'military'],
            'POST_REVOLUTION': ['count', 'noble', 'after war']
        }
    }
    
    # Extract years from claim
    years = re.findall(r'\b(17|18|19)\d{2}\b', claim_text)
    
    # Simple timeline violation check
    if character_lower in timeline_bins:
        bins = timeline_bins[character_lower]
        matched_bins = []
        
        for bin_name, markers in bins.items():
            if any(marker in claim_lower for marker in markers):
                matched_bins.append(bin_name)
        
        # Check for incompatible bins
        incompatible_pairs = [
            ('PRE_REVOLUTION', 'POST_REVOLUTION'),
            ('IMPRISONMENT', 'POST_ESCAPE'),
            ('REVOLUTIONARY', 'EXILE')
        ]
        
        for bin1, bin2 in incompatible_pairs:
            if bin1 in matched_bins and bin2 in matched_bins:
                return True
    
    return False


def aggregate_final_decision(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], semantic_index: SemanticIndex = None) -> Dict[str, str]:
    """
    Multi-stage evaluation with absence penalty, salience weighting, and asymmetric defaults.
    
    Args:
        claim: Single claim to evaluate
        evidence_chunks: Retrieved evidence chunks
        semantic_index: Semantic index for entropy estimation only
    
    Returns:
        Final decision with transparency logging
    """
    
    # Change 1: Classify claim salience (with downgrade heuristic)
    claim_salience = classify_claim_salience(claim.get('claim_text', ''))
    
    # Quick heuristic: downgrade HIGH salience if no strong atoms expected
    atoms = decompose_claim(claim.get('claim_text', ''))
    atoms = atoms[:7]  # Limit to max 7 atoms
    max_atom_weight = max([classify_atom_weight(atom) for atom in atoms]) if atoms else 0.5
    
    if claim_salience == "HIGH" and max_atom_weight < 1.5:
        claim_salience = "LOW"
    
    # Change 3: Check timeline violations
    character = claim.get('character', '')
    timeline_violation = check_timeline_violation(claim.get('claim_text', ''), character)
    
    if timeline_violation:
        return {
            "final_decision": "contradict",
            "explanation": "TIMELINE_VIOLATION: Claim spans incompatible timeline bins",
            "grounded_verdict": "TIMELINE_VIOLATION",
            "semantic_verdict": None,
            "method": "TIMELINE_VIOLATION",
            "claim_salience": claim_salience,
            "absence_penalty": 0,
            "weighted_conflict_score": 0.0,
            "decision_reason": "timeline_violation"
        }
    
    # Step 1: Tightened Over-Specification Detection
    claim_text = claim.get('claim_text', '').lower()
    
    # Count specificity indicators (named entities, dates, locations, numbers)
    specificity_indicators = [
        # Named entities and proper nouns
        'ritual', 'ceremony', 'ceremonial', 'symbolic', 'symbolism',
        'secret rites', 'named secret', 'invented rituals',
        'society', 'organization', 'meeting', 'congress', 'salon',
        'conspiracy', 'plot', 'leader', 'head of', 'founded', 'established',
        # Specific dates, numbers, locations
        'years old', 'age of', 'born in', 'died in', 'lived in',
        'exactly', 'precisely', 'specifically', 'particular'
    ]
    
    # Count specificity markers
    specificity_count = sum(1 for indicator in specificity_indicators if indicator in claim_text)
    
    # Constants for over-specification heuristic
    HIGH_SPECIFICITY_THRESHOLD = 2  # Number of specificity indicators
    LOW_EVIDENCE_THRESHOLD = 3      # Number of evidence chunks
    
    is_high_specificity = specificity_count >= HIGH_SPECIFICITY_THRESHOLD
    is_low_evidence = len(evidence_chunks) < LOW_EVIDENCE_THRESHOLD
    
    # Step 2: Decompose claim into atoms (already done above for salience check)
    print(f"Decomposing claim...")
    print(f"Got {len(atoms)} atoms")
    
    # Step 3: Evaluate each atom independently with weights and strong support tracking
    hard_violations = []
    weak_conflicts = []
    unsupported = []
    no_constraints = []
    supported = []
    atom_weights = []
    
    # Track strong support
    strong_supported = 0
    strong_total = 0
    
    for i, atom in enumerate(atoms):
        print(f"Evaluating atom {i+1}/{len(atoms)}: {atom[:30]}...")
        atom_claim = {**claim, 'claim_text': atom}
        
        # Add delay to prevent rate limits
        time.sleep(2)
        
        verdict = grounded_constraint_inference(atom_claim, evidence_chunks)
        weight = classify_atom_weight(atom)
        atom_weights.append(weight)
        
        # Track strong atoms
        if is_strong_expected_atom(atom, weight):
            strong_total += 1
            if verdict == "SUPPORTED":
                strong_supported += 1
        
        print(f"Atom {i+1} result: {verdict} (weight: {weight})")
        
        if verdict == "SUPPORTED":
            supported.append((atom, weight))
        elif verdict == "HARD_VIOLATION":
            hard_violations.append((atom, weight))
        elif verdict == "WEAK_CONFLICT":
            weak_conflicts.append((atom, weight))
        elif verdict == "UNSUPPORTED":
            unsupported.append((atom, weight))
        else:
            no_constraints.append((atom, weight))
    
    # Change 1: Improved absence penalty calculation
    retrieved_chunks = len(evidence_chunks)
    supported_atoms = len(supported) + len(no_constraints)  # Both SUPPORTED and NO_CONSTRAINT count as supported
    absence_penalty = 0
    
    if (claim_salience == "HIGH" and retrieved_chunks > 0 and 
        (supported_atoms == 0 or max_atom_weight < 1.5)):
        absence_penalty = 1
        # Add a weak conflict for absence
        weak_conflicts.append(("ABSENCE_PENALTY: High-salience claim with no strong supporting atoms", 1.0))
    
    # Change 5: Expected-Evidence Violation
    expected_evidence_violation = False
    if claim_salience == "HIGH" and strong_total > 0 and strong_supported == 0:
        expected_evidence_violation = True
    
    # Change 6: Covert keywords check
    covert_keywords_present = contains_covert_keywords(claim.get('claim_text', ''))
    covert_violation = False
    if (claim_salience == "HIGH" and covert_keywords_present and 
        len(supported) == 0):  # No directly supported atoms
        covert_violation = True
    
    # Change 2: Salience-weighted conflict voting
    weighted_conflict_score = sum(weight for _, weight in weak_conflicts)
    
    # Step 4: Apply decision hierarchy with correct order
    
    # Rule 1: HARD_VIOLATION → CONTRADICT
    if hard_violations:
        return {
            "final_decision": "contradict",
            "explanation": f"HARD_VIOLATION: {hard_violations[0][0][:50]}...",
            "grounded_verdict": "HARD_VIOLATION",
            "semantic_verdict": None,
            "method": "HARD_VIOLATION",
            "atoms_evaluated": len(atoms),
            "violations": len(hard_violations),
            "weak_conflicts": len(weak_conflicts),
            "claim_salience": claim_salience,
            "absence_penalty": absence_penalty,
            "weighted_conflict_score": weighted_conflict_score,
            "decision_reason": "hard_violation"
        }
    
    # Rule 2: TIMELINE_VIOLATION → CONTRADICT
    # (Already handled above)
    
    # Rule 3: Weighted conflict score >= 2.0 → CONTRADICT
    if weighted_conflict_score >= 2.0:
        return {
            "final_decision": "contradict",
            "explanation": f"WEIGHTED_CONFLICT: Score {weighted_conflict_score:.1f} >= 2.0",
            "grounded_verdict": "WEIGHTED_CONFLICT",
            "semantic_verdict": None,
            "method": "WEIGHTED_CONFLICT",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "weak_conflicts": len(weak_conflicts),
            "claim_salience": claim_salience,
            "absence_penalty": absence_penalty,
            "weighted_conflict_score": weighted_conflict_score,
            "decision_reason": "weighted_conflict_high"
        }
    
    # Rule 4: Weighted conflict score >= 1.0 AND absence penalty → CONTRADICT
    if weighted_conflict_score >= 1.0 and absence_penalty == 1:
        return {
            "final_decision": "contradict",
            "explanation": f"WEIGHTED_CONFLICT_WITH_ABSENCE: Score {weighted_conflict_score:.1f} + absence penalty",
            "grounded_verdict": "WEIGHTED_CONFLICT_WITH_ABSENCE",
            "semantic_verdict": None,
            "method": "WEIGHTED_CONFLICT_WITH_ABSENCE",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "weak_conflicts": len(weak_conflicts),
            "claim_salience": claim_salience,
            "absence_penalty": absence_penalty,
            "weighted_conflict_score": weighted_conflict_score,
            "decision_reason": "weighted_conflict_with_absence"
        }
    
    # Rule 5: Expected-Evidence Violation → CONTRADICT
    if expected_evidence_violation:
        return {
            "final_decision": "contradict",
            "explanation": f"EXPECTED_EVIDENCE_VIOLATION: High-salience claim with {strong_total} strong atoms, {strong_supported} supported",
            "grounded_verdict": "EXPECTED_EVIDENCE_VIOLATION",
            "semantic_verdict": None,
            "method": "EXPECTED_EVIDENCE_VIOLATION",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "weak_conflicts": len(weak_conflicts),
            "claim_salience": claim_salience,
            "absence_penalty": absence_penalty,
            "weighted_conflict_score": weighted_conflict_score,
            "strong_total": strong_total,
            "strong_supported": strong_supported,
            "decision_reason": "expected_evidence_violation"
        }
    
    # Rule 6: Covert Keywords Violation → CONTRADICT
    if covert_violation:
        return {
            "final_decision": "contradict",
            "explanation": f"COVERT_KEYWORDS_VIOLATION: High-salience claim with covert keywords but no support",
            "grounded_verdict": "COVERT_KEYWORDS_VIOLATION",
            "semantic_verdict": None,
            "method": "COVERT_KEYWORDS_VIOLATION",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "weak_conflicts": len(weak_conflicts),
            "claim_salience": claim_salience,
            "absence_penalty": absence_penalty,
            "weighted_conflict_score": weighted_conflict_score,
            "decision_reason": "covert_keywords_violation"
        }
    
    # Rule 7: Absence penalty AND salience == HIGH → CONTRADICT (gated version)
    if absence_penalty == 1 and claim_salience == "HIGH" and (len(weak_conflicts) > 0):
        return {
            "final_decision": "contradict",
            "explanation": f"ABSENCE_WITH_HIGH_SALIENCE: High-salience claim with absence penalty and weak conflicts",
            "grounded_verdict": "ABSENCE_WITH_HIGH_SALIENCE",
            "semantic_verdict": None,
            "method": "ABSENCE_WITH_HIGH_SALIENCE",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "weak_conflicts": len(weak_conflicts),
            "claim_salience": claim_salience,
            "absence_penalty": absence_penalty,
            "weighted_conflict_score": weighted_conflict_score,
            "decision_reason": "absence_with_high_salience"
        }
    
    # Rule 8: OVER_SPECIFIED → CONTRADICT (high specificity + low evidence + no hard constraints)
    if is_high_specificity and is_low_evidence and not hard_violations:
        return {
            "final_decision": "contradict",
            "explanation": f"OVER_SPECIFIED: High specificity ({specificity_count}) with low evidence ({len(evidence_chunks)})",
            "grounded_verdict": "OVER_SPECIFIED",
            "semantic_verdict": None,
            "method": "OVER_SPECIFIED",
            "atoms_evaluated": len(atoms),
            "violations": 0,
            "weak_conflicts": len(weak_conflicts),
            "claim_salience": claim_salience,
            "absence_penalty": absence_penalty,
            "weighted_conflict_score": weighted_conflict_score,
            "decision_reason": "over_specified"
        }
    
    # Rule 9: Default to CONSISTENT (both HIGH and LOW salience)
    # Note: Semantics may be used for entropy estimation but NOT for final verdict
    semantic_entropy = None
    if semantic_index and (unsupported or weak_conflicts):
        # Use semantics only for entropy estimation, not verdict
        semantic_chunks = semantic_index.semantic_neighborhood_retrieve(claim, top_k=10)
        semantic_verdict = semantic_neighborhood_evaluation(claim, semantic_chunks)
        semantic_entropy = "HIGH" if semantic_verdict == "INCOMPATIBLE" else "LOW"
    
    return {
        "final_decision": "consistent",
        "explanation": f"NO_CONSTRAINT: No rule-based violations found (salience: {claim_salience}, entropy: {semantic_entropy})",
        "grounded_verdict": "NO_CONSTRAINT",
        "semantic_verdict": None,  # Semantics not used for verdict
        "method": "NO_CONSTRAINT_CONSISTENT",
        "atoms_evaluated": len(atoms),
        "violations": 0,
        "weak_conflicts": len(weak_conflicts),
        "semantic_entropy": semantic_entropy,
        "claim_salience": claim_salience,
        "absence_penalty": absence_penalty,
        "weighted_conflict_score": weighted_conflict_score,
        "strong_total": strong_total,
        "strong_supported": strong_supported,
        "decision_reason": "no_constraint_consistent"
    }


# Legacy function for backward compatibility
def aggregate_final_decision_full(constraint_results: List[Dict[str, str]], book_context: str = "", original_claims: List[Dict[str, str]] = None, all_chunks: List[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Legacy function - deprecated in favor of new two-stage system.
    """
    if not constraint_results:
        return {
            "decision": "INCONSISTENT",
            "summary": "No claims to evaluate",
            "supporting_claims": [],
            "problematic_claims": []
        }
    
    # Simple aggregation for backward compatibility
    supported = [r for r in constraint_results if r.get('constraint_type') in ['supported', 'weakly_supported']]
    contradicted = [r for r in constraint_results if r.get('constraint_type') in ['contradicted', 'incompatible']]
    
    if contradicted:
        return {
            "decision": "INCONSISTENT",
            "summary": f"Claims contradicted ({len(contradicted)} contradicted)",
            "supporting_claims": [c['claim_id'] for c in supported],
            "problematic_claims": [c['claim_id'] for c in contradicted]
        }
    elif supported:
        return {
            "decision": "CONSISTENT",
            "summary": f"Claims supported ({len(supported)} supported)",
            "supporting_claims": [c['claim_id'] for c in supported],
            "problematic_claims": []
        }
    else:
        return {
            "decision": "INCONSISTENT",
            "summary": "No supporting evidence found",
            "supporting_claims": [],
            "problematic_claims": [c['claim_id'] for c in constraint_results]
        }