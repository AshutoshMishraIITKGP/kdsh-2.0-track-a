from typing import List, Dict
import time
from claim_decomposer import decompose_claim
from grounded_inference import grounded_constraint_inference
from semantic_neighborhood import semantic_neighborhood_evaluation
from semantic_index import SemanticIndex


def aggregate_final_decision(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], semantic_index: SemanticIndex = None) -> Dict[str, str]:
    """
    Multi-stage evaluation: Atom decomposition → Grounded → Semantic fallback
    
    Args:
        claim: Single claim to evaluate
        evidence_chunks: Retrieved evidence chunks
        semantic_index: Semantic index for neighborhood retrieval
    
    Returns:
        Final decision with transparency logging
    """
    
    # Step 1: Check for OVER_SPECIFICATION first (before atom evaluation)
    claim_text = claim.get('claim_text', '').lower()
    over_spec_indicators = [
        'ritual', 'ceremony', 'ceremonial', 'symbolic', 'symbolism',
        'secret rites', 'named secret', 'invented rituals',
        'society', 'organization', 'meeting', 'congress', 'salon',
        'conspiracy', 'plot', 'leader', 'head of', 'founded', 'established'
    ]
    
    is_over_specified = any(indicator in claim_text for indicator in over_spec_indicators)
    
    if is_over_specified:
        return {
            "final_decision": "contradict",
            "explanation": "OVER_SPECIFIED: Introduces specific details without textual anchoring",
            "grounded_verdict": "OVER_SPECIFIED",
            "semantic_verdict": None,
            "method": "OVER_SPECIFIED",
            "atoms_evaluated": 0,
            "violations": 1
        }
    
    # Step 2: Decompose claim into atoms (3-7 atoms)
    print(f"Decomposing claim...")
    atoms = decompose_claim(claim.get('claim_text', ''))
    atoms = atoms[:7]  # Limit to max 7 atoms
    print(f"Got {len(atoms)} atoms")
    
    # Step 3: Evaluate each atom independently
    hard_violations = []
    unsupported = []
    no_constraints = []
    
    for i, atom in enumerate(atoms):
        print(f"Evaluating atom {i+1}/{len(atoms)}: {atom[:30]}...")
        atom_claim = {**claim, 'claim_text': atom}
        
        # Add delay to prevent rate limits
        time.sleep(2)
        
        verdict = grounded_constraint_inference(atom_claim, evidence_chunks)
        print(f"Atom {i+1} result: {verdict}")
        
        if verdict == "HARD_VIOLATION":
            hard_violations.append(atom)
        elif verdict == "UNSUPPORTED":
            unsupported.append(atom)
        else:
            no_constraints.append(atom)
    
    # Step 4: Apply decision rules in exact order
    
    # Rule 1: HARD_VIOLATION overrides everything
    if hard_violations:
        return {
            "final_decision": "contradict",
            "explanation": f"HARD_VIOLATION: {hard_violations[0][:50]}...",
            "grounded_verdict": "HARD_VIOLATION",
            "semantic_verdict": None,
            "method": "HARD_VIOLATION",
            "atoms_evaluated": len(atoms),
            "violations": len(hard_violations)
        }
    
    # Rule 2: Classify claim impact
    # LOW_IMPACT indicators (childhood color, symbolic rituals, emotional descriptions, habits)
    low_impact_keywords = [
        'loved', 'admired', 'respected', 'feared', 'trusted', 'briefly',
        'personal', 'feeling', 'emotional', 'private', 'moment', 'childhood',
        'symbolic', 'ritual', 'habit', 'quirk', 'color', 'description'
    ]
    
    # CAUSAL_IMPACT indicators (changes alliances, alters motivations, explains actions, rewrites events)
    causal_impact_keywords = [
        'alliance', 'betrayal', 'motivation', 'explains', 'caused', 'led to',
        'resulted in', 'changed', 'altered', 'influenced', 'shaped', 'drove',
        'political', 'conspiracy', 'plot', 'war', 'revolution'
    ]
    
    is_low_impact = any(kw in claim_text for kw in low_impact_keywords)
    is_causal_impact = any(kw in claim_text for kw in causal_impact_keywords)
    
    # Rule 3: Handle UNSUPPORTED claims based on impact
    if unsupported:
        if is_low_impact:
            return {
                "final_decision": "consistent",
                "explanation": "UNSUPPORTED_LOW_IMPACT: Unsupported but low-impact claim",
                "grounded_verdict": "UNSUPPORTED",
                "semantic_verdict": None,
                "method": "UNSUPPORTED_LOW_IMPACT",
                "atoms_evaluated": len(atoms),
                "violations": 0
            }
        elif is_causal_impact and semantic_index:
            # Use semantic only for causal impact claims
            semantic_chunks = semantic_index.semantic_neighborhood_retrieve(claim, top_k=20)
            semantic_verdict = semantic_neighborhood_evaluation(claim, semantic_chunks)
            
            if semantic_verdict == "INCOMPATIBLE":
                return {
                    "final_decision": "contradict",
                    "explanation": "UNSUPPORTED_CAUSAL_SEMANTIC: Causal claim with narrative incompatibility",
                    "grounded_verdict": "UNSUPPORTED",
                    "semantic_verdict": semantic_verdict,
                    "method": "UNSUPPORTED_CAUSAL_SEMANTIC",
                    "atoms_evaluated": len(atoms),
                    "violations": 0
                }
            else:
                return {
                    "final_decision": "consistent",
                    "explanation": "UNSUPPORTED_CAUSAL_SEMANTIC: Causal claim with narrative compatibility",
                    "grounded_verdict": "UNSUPPORTED",
                    "semantic_verdict": semantic_verdict,
                    "method": "UNSUPPORTED_CAUSAL_SEMANTIC",
                    "atoms_evaluated": len(atoms),
                    "violations": 0
                }
    
    # Rule 4: Default for NO_CONSTRAINT atoms
    return {
        "final_decision": "consistent",
        "explanation": "NO_CONSTRAINT: No violations or constraints found",
        "grounded_verdict": "NO_CONSTRAINT",
        "semantic_verdict": None,
        "method": "NO_CONSTRAINT_CONSISTENT",
        "atoms_evaluated": len(atoms),
        "violations": 0
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