#!/usr/bin/env python3
"""
Test Novelty 6 - Event Salience × Canonical Silence
"""

import sys
sys.path.append('src')

from final_decision import check_event_footprint_silence, classify_atom_semantic_type

def test_novelty_6_event_footprint():
    """Test Novelty 6: Event Salience × Canonical Silence"""
    print("=== NOVELTY 6: Event Salience × Canonical Silence ===")
    
    # Test atoms with expected footprints
    high_salience_events = [
        "John met Napoleon Bonaparte in Paris",
        "He was recruited by the Secret Society", 
        "The character joined the underground circle",
        "He participated in the mutiny against Captain Smith",
        "John was arrested for betrayal"
    ]
    
    # Test atoms without expected footprints
    normal_events = [
        "John walked to the market",
        "He ate breakfast",
        "The weather was cold"
    ]
    
    # Empty evidence (no footprint)
    empty_evidence = []
    
    # Evidence with some footprints
    evidence_with_footprint = [
        {"text": "Napoleon was known to meet many people in Paris"},
        {"text": "There were several arrests during that period"}
    ]
    
    # Test atom verdicts (no SUPPORTED or HARD_VIOLATION)
    no_support_verdicts = {
        "John met Napoleon Bonaparte in Paris": "UNSUPPORTED",
        "He was recruited by the Secret Society": "UNSUPPORTED",
        "The character joined the underground circle": "NO_CONSTRAINT"
    }
    
    # Test atom verdicts with SUPPORTED (should prevent detection)
    with_support_verdicts = {
        "John met Napoleon Bonaparte in Paris": "SUPPORTED",
        "He was recruited by the Secret Society": "UNSUPPORTED"
    }
    
    print("Testing event footprint silence detection:")
    
    # Test 1: High-salience events with no footprint should be detected
    silent_events = check_event_footprint_silence(
        high_salience_events, empty_evidence, "HIGH", no_support_verdicts
    )
    print(f"Silent events detected (no footprint): {len(silent_events)}")
    for event in silent_events:
        print(f"  - {event}")
    
    # Test 2: Normal events should not be detected
    silent_normal = check_event_footprint_silence(
        normal_events, empty_evidence, "HIGH", {}
    )
    print(f"Normal events detected as silent: {len(silent_normal)}")
    
    # Test 3: LOW salience should not trigger detection
    silent_low_salience = check_event_footprint_silence(
        high_salience_events, empty_evidence, "LOW", no_support_verdicts
    )
    print(f"Silent events detected (LOW salience): {len(silent_low_salience)}")
    
    # Test 4: SUPPORTED atoms should prevent detection
    silent_with_support = check_event_footprint_silence(
        high_salience_events, empty_evidence, "HIGH", with_support_verdicts
    )
    print(f"Silent events detected (with SUPPORTED): {len(silent_with_support)}")
    
    # Test 5: Evidence with footprint should reduce detection
    silent_with_footprint = check_event_footprint_silence(
        high_salience_events, evidence_with_footprint, "HIGH", no_support_verdicts
    )
    print(f"Silent events detected (with footprint): {len(silent_with_footprint)}")
    
    # Assertions
    assert len(silent_events) > 0, "Should detect silent events with no footprint"
    assert len(silent_normal) == 0, "Should not detect normal events as silent"
    assert len(silent_low_salience) == 0, "Should not detect with LOW salience"
    assert len(silent_with_support) == 0, "Should not detect when SUPPORTED atoms exist"
    assert len(silent_with_footprint) < len(silent_events), "Evidence footprint should reduce detection"
    
    print("[OK] Event Salience × Canonical Silence works correctly")


def test_footprint_patterns():
    """Test specific event footprint patterns"""
    print("\n=== Event Footprint Pattern Detection ===")
    
    test_cases = [
        ("John met Napoleon in Paris", True, "named person interaction"),
        ("He was recruited by the Brotherhood", True, "named organization interaction"),
        ("The character joined the secret society", True, "named organization"),
        ("John participated in the mutiny", True, "major action - mutiny"),
        ("He was arrested for betrayal", True, "major action - arrest"),
        ("John walked to the market", False, "normal activity"),
        ("He felt happy", False, "emotion")
    ]
    
    empty_evidence = []
    no_support_verdicts = {}
    
    for atom, should_detect, description in test_cases:
        # Check if it's an EVENT atom first
        is_event = classify_atom_semantic_type(atom) == 'EVENT'
        
        if is_event:
            detected = check_event_footprint_silence([atom], empty_evidence, "HIGH", no_support_verdicts)
            has_silent = len(detected) > 0
            
            print(f"'{atom}' -> {has_silent} (expected: {should_detect}) [{description}]")
            
            if should_detect:
                assert has_silent, f"Should detect silent event: {atom}"
            else:
                assert not has_silent, f"Should not detect normal event: {atom}"
        else:
            print(f"'{atom}' -> Not EVENT type [{description}]")
    
    print("[OK] Event footprint patterns work correctly")


if __name__ == "__main__":
    print("Testing Novelty 6 - Event Salience × Canonical Silence\n")
    
    test_novelty_6_event_footprint()
    test_footprint_patterns()
    
    print("\n[SUCCESS] Novelty 6 implemented and tested successfully!")
    print("\nEvent Salience × Canonical Silence:")
    print("- Detects high-salience events with missing expected narrative footprints")
    print("- Only triggers when no SUPPORTED or HARD_VIOLATION atoms exist")
    print("- Checks for named interactions, organizations, and major actions")
    print("- Uses narrative necessity logic (not absence of evidence)")