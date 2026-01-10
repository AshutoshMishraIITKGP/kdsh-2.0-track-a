#!/usr/bin/env python3
"""
Test Scoped Ensemble Logic
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from final_decision import classify_claim_type, compute_salience


def test_scoped_ensemble():
    """Test scoped ensemble routing logic."""
    
    print("=== Scoped Ensemble Routing Test ===\n")
    
    # Cases that should use BASE MODEL (no ensemble)
    base_model_cases = [
        "He had a gentle personality",
        "She believed in justice", 
        "The character was known for wisdom",
        "He maintained his dignity",
        "She possessed great courage"
    ]
    
    print("Cases using BASE MODEL (no ensemble):")
    for claim in base_model_cases:
        claim_type = classify_claim_type(claim)
        salience = compute_salience(claim)
        use_base = (claim_type == "CANONICAL_BACKGROUND" and salience == "LOW")
        status = "BASE_MODEL" if use_base else "ENSEMBLE"
        print(f"  {status}: '{claim}' -> Type: {claim_type}, Salience: {salience}")
    
    print()
    
    # Cases that should use ENSEMBLE
    ensemble_cases = [
        "He secretly conspired against the government",  # HIGH salience
        "The character met with revolutionary leaders",  # SUPPORTED_EVENT
        "She hid manuscripts in a secret compartment",  # FABRICATED_INVENTION
        "He plotted the major betrayal"  # HIGH salience CANONICAL_BACKGROUND
    ]
    
    print("Cases using ENSEMBLE:")
    for claim in ensemble_cases:
        claim_type = classify_claim_type(claim)
        salience = compute_salience(claim)
        use_base = (claim_type == "CANONICAL_BACKGROUND" and salience == "LOW")
        status = "BASE_MODEL" if use_base else "ENSEMBLE"
        print(f"  {status}: '{claim}' -> Type: {claim_type}, Salience: {salience}")
    
    print()
    print("Expected behavior:")
    print("- CANONICAL_BACKGROUND + LOW salience -> BASE MODEL (consistent)")
    print("- All other cases -> ENSEMBLE (three perspectives)")
    print("- This should recover ~10-15% accuracy by not over-processing simple cases")


if __name__ == "__main__":
    test_scoped_ensemble()