#!/usr/bin/env python3
"""
Test Novelty 5 - Canonical Plausibility Gate
"""

import sys
sys.path.append('src')

from final_decision import detect_fabricated_event_anchors, classify_atom_semantic_type

def test_novelty_5_fabricated_events():
    """Test Novelty 5: Canonical Plausibility Gate"""
    print("=== NOVELTY 5: Canonical Plausibility Gate ===")
    
    # Test atoms with fabricated event anchors
    fabricated_atoms = [
        "John founded the Secret Brotherhood of the Phoenix",
        "He performed the ancient Moonlight Oath ceremony", 
        "The character signed the Hidden Manifesto of Rebellion",
        "This was the turning point that changed everything"
    ]
    
    # Test atoms without fabricated anchors
    normal_atoms = [
        "John met Napoleon in Paris",
        "He traveled to London",
        "The battle was fierce"
    ]
    
    # Empty evidence (no structural footprint)
    empty_evidence = []
    
    # Evidence with some structure
    structured_evidence = [
        {"text": "There were many organizations in the city"},
        {"text": "Various ceremonies took place during festivals"}
    ]
    
    print("Testing fabricated event detection with HIGH salience:")
    
    # Test 1: Fabricated events with no evidence should be detected
    detected_fabricated = detect_fabricated_event_anchors(fabricated_atoms, empty_evidence, "HIGH")
    print(f"Fabricated events detected (empty evidence): {len(detected_fabricated)}")
    for event in detected_fabricated:
        print(f"  - {event}")
    
    # Test 2: Normal events should not be detected as fabricated
    detected_normal = detect_fabricated_event_anchors(normal_atoms, empty_evidence, "HIGH")
    print(f"Normal events detected as fabricated: {len(detected_normal)}")
    
    # Test 3: LOW salience should not trigger detection
    detected_low_salience = detect_fabricated_event_anchors(fabricated_atoms, empty_evidence, "LOW")
    print(f"Fabricated events detected (LOW salience): {len(detected_low_salience)}")
    
    # Test 4: Structured evidence should prevent detection
    detected_with_structure = detect_fabricated_event_anchors(fabricated_atoms, structured_evidence, "HIGH")
    print(f"Fabricated events detected (with structure): {len(detected_with_structure)}")
    
    # Assertions
    assert len(detected_fabricated) > 0, "Should detect fabricated events with no evidence"
    assert len(detected_normal) == 0, "Should not detect normal events as fabricated"
    assert len(detected_low_salience) == 0, "Should not detect with LOW salience"
    assert len(detected_with_structure) < len(detected_fabricated), "Structured evidence should reduce detection"
    
    print("[OK] Canonical Plausibility Gate works correctly")


def test_event_anchor_patterns():
    """Test specific fabricated event anchor patterns"""
    print("\n=== Event Anchor Pattern Detection ===")
    
    test_cases = [
        ("John joined the Secret Society of Masons", True, "named organization"),
        ("He performed the Blood Oath ritual", True, "named ritual"), 
        ("The character signed the Revolutionary Manifesto", True, "named document"),
        ("This was the turning point in his life", True, "turning point framing"),
        ("John walked to the market", False, "normal event"),
        ("He felt happy about the news", False, "emotion")
    ]
    
    for atom, should_detect, description in test_cases:
        # Check if it's an EVENT atom first
        is_event = classify_atom_semantic_type(atom) == 'EVENT'
        
        # Only EVENT atoms can be fabricated events
        if is_event:
            detected = detect_fabricated_event_anchors([atom], [], "HIGH")
            has_fabricated = len(detected) > 0
            
            print(f"'{atom}' -> {has_fabricated} (expected: {should_detect}) [{description}]")
            
            if should_detect:
                assert has_fabricated, f"Should detect fabricated event: {atom}"
            else:
                assert not has_fabricated, f"Should not detect normal event: {atom}"
        else:
            print(f"'{atom}' -> Not EVENT type [{description}]")
    
    print("[OK] Event anchor patterns work correctly")


if __name__ == "__main__":
    print("Testing Novelty 5 - Canonical Plausibility Gate\n")
    
    test_novelty_5_fabricated_events()
    test_event_anchor_patterns()
    
    print("\n[SUCCESS] Novelty 5 implemented and tested successfully!")
    print("\nCanonical Plausibility Gate:")
    print("- Detects fabricated event anchors with no structural footprint")
    print("- Only triggers on HIGH salience EVENT atoms") 
    print("- Checks for named organizations, rituals, documents, turning points")
    print("- Uses negative grounding (absence of structure logic)")