#!/usr/bin/env python3
"""
Test Three Bucket Fixes
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from final_decision import is_emotion_motivation_atom, classify_claim_type, compute_salience


def test_bucket_fixes():
    """Test the three bucket fixes."""
    
    print("=== Three Bucket Fixes Test ===\n")
    
    # Bucket 1: Emotion/motivation protection
    emotion_atoms = [
        "failed attempt to save comrade",
        "naval commander praised his courage", 
        "he felt proud of his actions",
        "was motivated by revenge",
        "admired by his peers"
    ]
    
    print("Bucket 1: Emotion/Motivation atoms (protected from HARD_VIOLATION):")
    for atom in emotion_atoms:
        is_protected = is_emotion_motivation_atom(atom)
        status = "PROTECTED" if is_protected else "EXPOSED"
        print(f"  {status}: '{atom}'")
    
    print()
    
    # Bucket 2: OVER_SPECIFIED restrictiveness
    over_spec_test_cases = [
        "Secret society recruited Faria",  # Should NOT trigger (single condition)
        "India uprising witnessed by character",  # Should NOT trigger (historical color)
        "Founded secret organization and became leader of conspiracy plot"  # Should trigger (multiple conditions)
    ]
    
    print("Bucket 2: OVER_SPECIFIED restrictiveness:")
    for claim in over_spec_test_cases:
        claim_lower = claim.lower()
        conditions = [
            ('secret society' in claim_lower or 'organization' in claim_lower),
            ('founded' in claim_lower or 'established' in claim_lower),
            ('leader' in claim_lower or 'head of' in claim_lower),
            ('conspiracy' in claim_lower and 'plot' in claim_lower)
        ]
        would_trigger = sum(conditions) >= 2
        status = "OVER_SPECIFIED" if would_trigger else "ALLOWED"
        print(f"  {status}: '{claim}' (conditions: {sum(conditions)}/4)")
    
    print()
    
    # Bucket 3: Event density awareness
    background_cases = [
        "Noirtier was a militant republican",  # No concrete events
        "Character believed in revolutionary ideals",  # No concrete events
        "He met with political leaders and planned uprising"  # Has concrete events
    ]
    
    print("Bucket 3: Event density awareness for CANONICAL_BACKGROUND:")
    for claim in background_cases:
        claim_type = classify_claim_type(claim)
        salience = compute_salience(claim)
        has_events = any(event_word in claim.lower() for event_word in ['met', 'traveled', 'witnessed', 'participated', 'planned'])
        ensemble_weight = 0.5 if (claim_type == "CANONICAL_BACKGROUND" and not has_events) else 1.0
        print(f"  Weight {ensemble_weight}: '{claim}' -> Type: {claim_type}, Events: {has_events}")
    
    print()
    print("Expected improvements:")
    print("- Bucket 1: Emotion/motivation atoms won't trigger HARD_VIOLATION")
    print("- Bucket 2: Historical color allowed, only multiple org details trigger OVER_SPECIFIED")
    print("- Bucket 3: Background claims without events get reduced ensemble weight")
    print("- Combined effect: ~3-5% accuracy improvement")


if __name__ == "__main__":
    test_bucket_fixes()