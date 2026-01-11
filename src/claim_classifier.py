"""
Claim Classification Module
Provides claim typing and salience estimation for expectation-aware reasoning.
"""

from typing import Dict, Tuple


def classify_claim_type(claim_text: str, atoms: list) -> str:
    """
    Classify claim into one of four types.
    
    Returns:
        CANONICAL_BACKGROUND: Background facts about character
        SUPPORTED_EVENT: Specific events that should be documented
        FABRICATED_INVENTION: Over-detailed ceremonial/organizational claims
        OVER_SPECIFIED: Excessive detail suggesting fabrication
    """
    claim_lower = claim_text.lower()
    
    # OVER_SPECIFIED: Fabricated rituals, secret societies, invented ceremonies
    over_spec_indicators = [
        'ritual', 'ceremony', 'secret society', 'secret order', 'sacred',
        'initiation', 'oath', 'brotherhood', 'sisterhood', 'covenant'
    ]
    if any(indicator in claim_lower for indicator in over_spec_indicators):
        return "OVER_SPECIFIED"
    
    # FABRICATED_INVENTION: Excessive organizational detail
    if len(atoms) > 5 and any(word in claim_lower for word in ['founded', 'established', 'organization']):
        return "FABRICATED_INVENTION"
    
    # SUPPORTED_EVENT: Public actions, arrests, meetings, alliances
    event_indicators = [
        'arrested', 'imprisoned', 'met', 'joined', 'allied', 'betrayed',
        'trial', 'escaped', 'confronted', 'visited', 'spoke with'
    ]
    if any(indicator in claim_lower for indicator in event_indicators):
        return "SUPPORTED_EVENT"
    
    # CANONICAL_BACKGROUND: Default for background facts
    return "CANONICAL_BACKGROUND"


def estimate_salience(claim_text: str, claim_type: str, atoms: list) -> str:
    """
    Estimate claim salience (LOW or HIGH).
    
    HIGH salience: Public events, major actions, documented facts
    LOW salience: Private thoughts, emotions, minor traits
    """
    claim_lower = claim_text.lower()
    
    # LOW salience indicators
    low_salience_patterns = [
        'felt', 'believed', 'thought', 'feared', 'hoped', 'admired',
        'loved', 'hated', 'respected', 'despised', 'childhood', 'boyhood',
        'family', 'mother', 'father', 'sibling', 'gathering', 'celebration'
    ]
    
    if any(pattern in claim_lower for pattern in low_salience_patterns):
        return "LOW"
    
    # HIGH salience for SUPPORTED_EVENT and OVER_SPECIFIED
    if claim_type in ["SUPPORTED_EVENT", "OVER_SPECIFIED"]:
        return "HIGH"
    
    # HIGH salience for canon-obligated content
    high_salience_patterns = [
        'arrested', 'imprisoned', 'trial', 'executed', 'escaped',
        'met', 'joined', 'founded', 'established', 'allied', 'betrayed'
    ]
    
    if any(pattern in claim_lower for pattern in high_salience_patterns):
        return "HIGH"
    
    # Default to LOW for background
    return "LOW"


def should_expect_evidence(claim_type: str, salience: str, atoms: list) -> bool:
    """
    Determine if evidence should be reasonably expected for this claim.
    
    Returns True only if:
    - salience == HIGH
    - claim_type in {SUPPORTED_EVENT, FABRICATED_INVENTION, CANONICAL_BACKGROUND}
    - claim implies public/canonical events
    """
    if salience != "HIGH":
        return False
    
    if claim_type not in ["SUPPORTED_EVENT", "FABRICATED_INVENTION", "CANONICAL_BACKGROUND"]:
        return False
    
    # Check if atoms contain public/canonical events
    public_event_indicators = [
        'arrested', 'imprisoned', 'trial', 'met', 'joined', 'founded',
        'allied', 'betrayed', 'escaped', 'confronted', 'visited'
    ]
    
    for atom in atoms:
        atom_lower = atom.lower()
        if any(indicator in atom_lower for indicator in public_event_indicators):
            return True
    
    return False


def classify_claim(claim_text: str, atoms: list) -> Dict[str, any]:
    """
    Complete claim classification pipeline.
    
    Returns:
        {
            'claim_type': str,
            'salience': str,
            'expected_evidence': bool
        }
    """
    claim_type = classify_claim_type(claim_text, atoms)
    salience = estimate_salience(claim_text, claim_type, atoms)
    expected_evidence = should_expect_evidence(claim_type, salience, atoms)
    
    return {
        'claim_type': claim_type,
        'salience': salience,
        'expected_evidence': expected_evidence
    }
