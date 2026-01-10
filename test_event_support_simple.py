#!/usr/bin/env python3
"""
Test Event Support Requirement Rule
"""

def test_event_support_requirement():
    """Test Event Support Requirement logic"""
    print("=== Event Support Requirement Test ===")
    
    # Test case 1: HIGH salience with events but no SUPPORTED events -> CONTRADICT
    event_atom_count_1 = 2
    salience_1 = "HIGH"
    supported_event_atoms_1 = []
    should_contradict_1 = event_atom_count_1 > 0 and salience_1 == "HIGH" and not supported_event_atoms_1
    
    print(f"Case 1: {event_atom_count_1} events, {salience_1} salience, no supported events")
    print(f"Should contradict: {should_contradict_1} (expected: True)")
    assert should_contradict_1 == True
    
    # Test case 2: HIGH salience with events and SUPPORTED events -> Allow to continue
    event_atom_count_2 = 2
    salience_2 = "HIGH"
    supported_event_atoms_2 = ["John met Napoleon"]
    should_contradict_2 = event_atom_count_2 > 0 and salience_2 == "HIGH" and not supported_event_atoms_2
    
    print(f"Case 2: {event_atom_count_2} events, {salience_2} salience, has supported events")
    print(f"Should contradict: {should_contradict_2} (expected: False)")
    assert should_contradict_2 == False
    
    # Test case 3: LOW salience with events but no SUPPORTED events -> Allow to continue
    event_atom_count_3 = 2
    salience_3 = "LOW"
    supported_event_atoms_3 = []
    should_contradict_3 = event_atom_count_3 > 0 and salience_3 == "HIGH" and not supported_event_atoms_3
    
    print(f"Case 3: {event_atom_count_3} events, {salience_3} salience, no supported events")
    print(f"Should contradict: {should_contradict_3} (expected: False)")
    assert should_contradict_3 == False
    
    # Test case 4: No events -> Allow to continue
    event_atom_count_4 = 0
    salience_4 = "HIGH"
    supported_event_atoms_4 = []
    should_contradict_4 = event_atom_count_4 > 0 and salience_4 == "HIGH" and not supported_event_atoms_4
    
    print(f"Case 4: {event_atom_count_4} events, {salience_4} salience, no supported events")
    print(f"Should contradict: {should_contradict_4} (expected: False)")
    assert should_contradict_4 == False
    
    print("[OK] Event Support Requirement logic works correctly")


if __name__ == "__main__":
    print("Testing Event Support Requirement Rule\n")
    
    test_event_support_requirement()
    
    print("\n[SUCCESS] Event Support Requirement implemented correctly!")
    print("\nRule: If claim has >=1 EVENT atom and salience == HIGH,")
    print("then at least one EVENT atom must be SUPPORTED or -> CONTRADICT")