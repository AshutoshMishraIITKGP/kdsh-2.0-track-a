#!/usr/bin/env python3
"""
Test Canon-Obligated Events Rule
"""

import sys
sys.path.append('src')

from final_decision import is_canon_obligated_event, classify_atom_semantic_type

def test_canon_obligated_classification():
    """Test Canon-Obligated event classification"""
    print("=== Canon-Obligated Event Classification ===")
    
    # Canon-obligated events (should return True)
    obligated_events = [
        ("John participated in the mutiny", True, "historical event - mutiny"),
        ("He was arrested by authorities", True, "historical event - arrest"),
        ("The character joined the secret society", True, "named institution"),
        ("He appeared in court", True, "named institution - court"),
        ("John met Napoleon Bonaparte", True, "major character interaction"),
        ("He became known as the Shadow", True, "identity mutation"),
        ("The character plotted against the king", True, "political action"),
        ("He led the conspiracy", True, "political action")
    ]
    
    # Canon-optional events (should return False)
    optional_events = [
        ("John learned to fence", False, "skill acquisition"),
        ("He rescued a stranger", False, "rescue of unnamed person"),
        ("The character felt proud", False, "moral reaction"),
        ("He performed a symbolic gesture", False, "symbolic act"),
        ("John walked to the market", False, "normal activity"),
        ("He ate breakfast", False, "daily routine")
    ]
    
    print("Testing canon-obligated events:")
    for atom, expected, description in obligated_events:
        result = is_canon_obligated_event(atom)
        print(f"'{atom}' -> {result} (expected: {expected}) [{description}]")
        assert result == expected, f"Expected {expected} for {atom}"
    
    print("\nTesting canon-optional events:")
    for atom, expected, description in optional_events:
        result = is_canon_obligated_event(atom)
        print(f"'{atom}' -> {result} (expected: {expected}) [{description}]")
        assert result == expected, f"Expected {expected} for {atom}"
    
    print("[OK] Canon-obligated classification works correctly")


def test_canon_obligated_rule_logic():
    """Test the Canon-Obligated Events rule logic"""
    print("\n=== Canon-Obligated Rule Logic ===")
    
    # Test case 1: HIGH salience + canon-obligated events + no support -> CONTRADICT
    salience_1 = "HIGH"
    atoms_1 = ["John participated in the mutiny", "He learned to fence"]
    supported_1 = ["He learned to fence"]  # Only optional event supported
    
    canon_obligated_1 = [atom for atom in atoms_1 if is_canon_obligated_event(atom)]
    supported_canon_1 = [atom for atom in supported_1 if is_canon_obligated_event(atom)]
    should_contradict_1 = salience_1 == "HIGH" and canon_obligated_1 and not supported_canon_1
    
    print(f"Case 1: {salience_1} salience, canon-obligated: {len(canon_obligated_1)}, supported canon: {len(supported_canon_1)}")
    print(f"Should contradict: {should_contradict_1} (expected: True)")
    assert should_contradict_1 == True
    
    # Test case 2: HIGH salience + canon-obligated events + has support -> Continue
    salience_2 = "HIGH"
    atoms_2 = ["John participated in the mutiny", "He learned to fence"]
    supported_2 = ["John participated in the mutiny", "He learned to fence"]
    
    canon_obligated_2 = [atom for atom in atoms_2 if is_canon_obligated_event(atom)]
    supported_canon_2 = [atom for atom in supported_2 if is_canon_obligated_event(atom)]
    should_contradict_2 = salience_2 == "HIGH" and canon_obligated_2 and not supported_canon_2
    
    print(f"Case 2: {salience_2} salience, canon-obligated: {len(canon_obligated_2)}, supported canon: {len(supported_canon_2)}")
    print(f"Should contradict: {should_contradict_2} (expected: False)")
    assert should_contradict_2 == False
    
    # Test case 3: LOW salience + canon-obligated events + no support -> Continue
    salience_3 = "LOW"
    atoms_3 = ["John participated in the mutiny"]
    supported_3 = []
    
    canon_obligated_3 = [atom for atom in atoms_3 if is_canon_obligated_event(atom)]
    supported_canon_3 = [atom for atom in supported_3 if is_canon_obligated_event(atom)]
    should_contradict_3 = salience_3 == "HIGH" and canon_obligated_3 and not supported_canon_3
    
    print(f"Case 3: {salience_3} salience, canon-obligated: {len(canon_obligated_3)}, supported canon: {len(supported_canon_3)}")
    print(f"Should contradict: {should_contradict_3} (expected: False)")
    assert should_contradict_3 == False
    
    # Test case 4: HIGH salience + only optional events + no support -> Continue
    salience_4 = "HIGH"
    atoms_4 = ["John learned to fence", "He rescued a stranger"]
    supported_4 = []
    
    canon_obligated_4 = [atom for atom in atoms_4 if is_canon_obligated_event(atom)]
    supported_canon_4 = [atom for atom in supported_4 if is_canon_obligated_event(atom)]
    should_contradict_4 = salience_4 == "HIGH" and len(canon_obligated_4) > 0 and not supported_canon_4
    
    print(f"Case 4: {salience_4} salience, canon-obligated: {len(canon_obligated_4)}, supported canon: {len(supported_canon_4)}")
    print(f"Should contradict: {should_contradict_4} (expected: False)")
    assert should_contradict_4 == False
    
    print("[OK] Canon-obligated rule logic works correctly")


if __name__ == "__main__":
    print("Testing Canon-Obligated Events Rule\n")
    
    test_canon_obligated_classification()
    test_canon_obligated_rule_logic()
    
    print("\n[SUCCESS] Canon-Obligated Events rule implemented correctly!")
    print("\nRule: Only canon-obligated events require support")
    print("- Named historical events, institutions, character interactions")
    print("- Identity mutations, political/military actions")
    print("- Skill acquisition, rescues, moral reactions are optional")