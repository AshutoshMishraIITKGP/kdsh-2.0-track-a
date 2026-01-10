from typing import List, Dict
import time
from claim_decomposer import decompose_claim
from grounded_inference import grounded_constraint_inference
from semantic_neighborhood import semantic_neighborhood_evaluation
from semantic_index import SemanticIndex
from ensemble_v1 import run_ensemble_decision


def is_emotion_motivation_atom(atom: str) -> bool:
    """Check if atom is emotion/motivation/social response type."""
    atom_lower = atom.lower()
    
    emotion_motivation_indicators = [
        # Emotions
        'felt', 'emotion', 'angry', 'sad', 'happy', 'proud', 'ashamed',
        # Motivations
        'wanted', 'hoped', 'intended', 'motivated', 'driven by',
        # Social responses
        'praised', 'criticized', 'admired', 'respected', 'honored',
        'failed attempt', 'naval praise', 'commander praised'
    ]
    
    return any(indicator in atom_lower for indicator in emotion_motivation_indicators)


def classify_atom_semantic_type(atom: str) -> str:
    """NOVELTY 3: Classify atom semantic type for historical texture allowance."""
    atom_lower = atom.lower()
    
    # Historical texture types that should not trigger CONTRADICT
    if any(word in atom_lower for word in ['ritual', 'ceremony', 'tradition']):
        return 'RITUAL'
    if any(word in atom_lower for word in ['felt', 'emotion', 'angry', 'sad', 'happy', 'proud']):
        return 'EMOTION'
    if any(word in atom_lower for word in ['symbol', 'represent', 'meaning', 'significance']):
        return 'SYMBOLIC'
    if any(word in atom_lower for word in ['habit', 'routine', 'custom', 'practice']):
        return 'HABIT'
    if any(word in atom_lower for word in ['thought', 'believed', 'considered', 'viewed']):
        return 'PSYCHOLOGICAL'
    if any(word in atom_lower for word in ['moral', 'ethical', 'right', 'wrong', 'virtue']):
        return 'MORAL_JUDGMENT'
    if any(word in atom_lower for word in ['background', 'family', 'origin', 'birth', 'childhood', 'born', 'hair', 'wealthy', 'poor', 'appearance', 'looks']):
        return 'BACKGROUND'
    
    return 'EVENT'  # Default to event type


def count_event_atoms(atoms: List[str]) -> int:
    """NOVELTY 1: Count atoms that represent concrete events."""
    event_count = 0
    for atom in atoms:
        atom_type = classify_atom_semantic_type(atom)
        if atom_type == 'EVENT':
            event_count += 1
    return event_count


def is_historical_texture_only(atoms: List[str]) -> bool:
    """NOVELTY 3: Check if all atoms are historical texture types."""
    texture_types = {'RITUAL', 'EMOTION', 'SYMBOLIC', 'HABIT', 'PSYCHOLOGICAL', 'MORAL_JUDGMENT', 'BACKGROUND'}
    
    for atom in atoms:
        atom_type = classify_atom_semantic_type(atom)
        if atom_type not in texture_types:
            return False
    return True


def is_named_character_event(atom: str) -> bool:
    """Check if atom involves named characters (proper nouns)."""
    words = atom.split()
    named_characters = 0
    
    for word in words:
        # Look for capitalized words that are likely proper names
        if (word[0].isupper() and 
            len(word) > 2 and 
            word.isalpha() and 
            word not in ['The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'With', 'By']):
            named_characters += 1
    
    return named_characters >= 1

def has_canon_confirmation(atom: str, evidence_chunks: List[Dict[str, str]]) -> bool:
    """Check if atom has canon confirmation in evidence."""
    atom_lower = atom.lower()
    evidence_text = " ".join([chunk.get('text', '') for chunk in evidence_chunks]).lower()
    
    # Extract key entities from the atom
    words = atom_lower.split()
    original_words = atom.split()
    key_entities = []
    
    # Look for proper names (capitalized words in original atom)
    for word in original_words:
        if (word[0].isupper() and 
            len(word) > 2 and 
            word.isalpha() and 
            word not in ['The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'With', 'By']):
            key_entities.append(word.lower())
    
    # Look for interaction verbs and their objects
    interaction_verbs = ['met', 'arrested', 'killed', 'rescued', 'encountered', 'spoke', 'betrayed', 'allied']
    for i, word in enumerate(words):
        if word in interaction_verbs and i + 1 < len(words):
            key_entities.append(words[i + 1])
    
    # Check for canon confirmation - require at least 2 entities mentioned together
    if len(key_entities) >= 2:
        entity_mentions = sum(1 for entity in key_entities if entity in evidence_text)
        
        if entity_mentions >= 2:
            # Check if they appear in proximity (same chunk)
            for chunk in evidence_chunks:
                chunk_text = chunk.get('text', '').lower()
                entities_in_chunk = [entity for entity in key_entities if entity in chunk_text]
                if len(entities_in_chunk) >= 2:
                    return True
    
def detect_fabricated_event_anchors(atoms: List[str], evidence_chunks: List[Dict[str, str]], salience: str) -> List[str]:
    """NOVELTY 5: Detect fabricated event anchors that have no structural footprint in canon."""
    if salience != "HIGH":
        return []
    
    fabricated_events = []
    evidence_text = " ".join([chunk.get('text', '') for chunk in evidence_chunks]).lower()
    
    for atom in atoms:
        if classify_atom_semantic_type(atom) != 'EVENT':
            continue
            
        atom_lower = atom.lower()
        
        # Check for fabricated event anchor patterns
        has_named_org = any(word in atom_lower for word in ['society', 'order', 'league', 'brotherhood', 'organization'])
        has_named_ritual = any(word in atom_lower for word in ['handshake', 'oath', 'ceremony', 'ritual', 'rite'])
        has_named_document = any(word in atom_lower for word in ['contract', 'decree', 'manifesto', 'charter', 'treaty'])
        has_turning_point = any(phrase in atom_lower for phrase in ['turning point', 'pivotal moment', 'decisive', 'changed everything'])
        
        if has_named_org or has_named_ritual or has_named_document or has_turning_point:
            # Check if evidence has ANY structural similarity
            has_structural_footprint = any(word in evidence_text for word in [
                'society', 'order', 'organization', 'group', 'brotherhood',
                'ceremony', 'ritual', 'oath', 'rite', 'tradition',
                'document', 'contract', 'decree', 'charter', 'letter',
                'moment', 'event', 'turning', 'change', 'decisive'
            ])
            
            if not has_structural_footprint:
                fabricated_events.append(atom)
    
    return fabricated_events
        for i, word in enumerate(words):
            if word in interaction_verbs and i + 1 < len(words):
                # Add the object of the interaction
                key_entities.append(words[i + 1])
        
        # Check for zero canon confirmation
        has_canon_confirmation = False
        if key_entities:
            # Look for any mention of the key entities together in evidence
            entity_mentions = sum(1 for entity in key_entities if entity in evidence_text)
            
            # Require at least 2 entities mentioned together for confirmation
            if entity_mentions >= 2:
                # Check if they appear in proximity (rough heuristic)
                for chunk in evidence_chunks:
                    chunk_text = chunk.get('text', '').lower()
                    entities_in_chunk = [entity for entity in key_entities if entity in chunk_text]
                    if len(entities_in_chunk) >= 2:
                        has_canon_confirmation = True
                        break
        
        # If zero canon confirmation for timeline-anchoring event
        if not has_canon_confirmation and key_entities:
            canon_missing_events.append(atom)
    
def detect_fabricated_event_anchors(atoms: List[str], evidence_chunks: List[Dict[str, str]], salience: str) -> List[str]:
    """NOVELTY 5: Detect fabricated event anchors that have no structural footprint in canon."""
    if salience != "HIGH":
        return []
    
    fabricated_events = []
    evidence_text = " ".join([chunk.get('text', '') for chunk in evidence_chunks]).lower()
    
    for atom in atoms:
        if classify_atom_semantic_type(atom) != 'EVENT':
            continue
            
        atom_lower = atom.lower()
        
        # Check for fabricated event anchor patterns
        has_named_org = any(word in atom_lower for word in ['society', 'order', 'league', 'brotherhood', 'organization'])
        has_named_ritual = any(word in atom_lower for word in ['handshake', 'oath', 'ceremony', 'ritual', 'rite'])
        has_named_document = any(word in atom_lower for word in ['contract', 'decree', 'manifesto', 'charter', 'treaty'])
        has_turning_point = any(phrase in atom_lower for phrase in ['turning point', 'pivotal moment', 'decisive', 'changed everything'])
        
        if has_named_org or has_named_ritual or has_named_document or has_turning_point:
            # Check if evidence has ANY structural similarity
            has_structural_footprint = any(word in evidence_text for word in [
                'society', 'order', 'organization', 'group', 'brotherhood',
                'ceremony', 'ritual', 'oath', 'rite', 'tradition',
                'document', 'contract', 'decree', 'charter', 'letter',
                'moment', 'event', 'turning', 'change', 'decisive'
            ])
            
            if not has_structural_footprint:
                fabricated_events.append(atom)
    
    return fabricated_events


def get_canon_event_tier(atom: str) -> int:
    """Classify canon events by criticality tier (3=highest, 1=lowest, 0=not canon event)."""
    atom_lower = atom.lower()
    
    # Tier 3: Identity locks - 1 failure is enough
    tier3_indicators = [
        'died', 'death', 'killed', 'murdered', 'executed',
        'imprisoned', 'prison', 'jail', 'cell',
        'first met', 'first meeting', 'initially met',
        'son of', 'daughter of', 'child of', 'parent', 'father', 'mother',
        'born in', 'birthplace', 'native of'
    ]
    
    # Tier 2: Timeline anchors - 2 failures needed
    tier2_indicators = [
        'arrested', 'arrest', 'captured',
        'exiled', 'exile', 'banished',
        'war', 'battle', 'siege', 'fought in',
        'married', 'wedding', 'divorce',
        'appointed', 'promoted', 'dismissed'
    ]
    
    # Tier 1: Contextual canon - never contradict alone
    tier1_indicators = [
        'member of', 'joined', 'left organization',
        'believed', 'supported', 'opposed',
        'visited', 'traveled to', 'lived in'
    ]
    
    for indicator in tier3_indicators:
        if indicator in atom_lower:
            return 3
    
    for indicator in tier2_indicators:
        if indicator in atom_lower:
            return 2
    
    for indicator in tier1_indicators:
        if indicator in atom_lower:
            return 1
    
    return 0

def is_canon_obligated_event(atom: str) -> bool:
    """Check if atom describes a canon-obligated event that requires textual support."""
    return get_canon_event_tier(atom) > 0

def detect_silent_canon_events(atoms: List[str], evidence_chunks: List[Dict[str, str]], salience: str, atom_verdicts: Dict[str, str]) -> List[str]:
    """NOVELTY 6: Check for high-salience events with missing expected narrative footprints."""
    if salience != "HIGH":
        return []
    
    # Only proceed if no atoms are explicitly SUPPORTED and no HARD_VIOLATION fired
    has_supported = any(verdict == "SUPPORTED" for verdict in atom_verdicts.values())
    has_hard_violation = any(verdict == "HARD_VIOLATION" for verdict in atom_verdicts.values())
    
    if has_supported or has_hard_violation:
        return []
    
    silent_events = []
    evidence_text = " ".join([chunk.get('text', '') for chunk in evidence_chunks]).lower()
    
    for atom in atoms:
        if classify_atom_semantic_type(atom) != 'EVENT':
            continue
            
        atom_lower = atom.lower()
        
        # Check for events with expected footprint
        has_named_interaction = any(word in atom_lower for word in ['met', 'recruited', 'joined', 'worked with', 'allied with'])
        has_named_organization = any(word in atom_lower for word in ['society', 'organization', 'circle', 'group', 'brotherhood'])
        has_major_action = any(word in atom_lower for word in ['mutiny', 'trial', 'arrest', 'betrayal', 'execution', 'rebellion'])
        
        if has_named_interaction or has_named_organization or has_major_action:
            # Extract key terms that should appear in evidence
            key_terms = []
            
            # Extract names and organizations
            words = atom_lower.split()
            for i, word in enumerate(words):
                if word in ['met', 'recruited', 'joined'] and i + 1 < len(words):
                    key_terms.append(words[i + 1])  # Person name after interaction verb
                elif word in ['society', 'organization', 'circle', 'group']:
                    if i > 0:
                        key_terms.append(words[i - 1])  # Adjective before organization
                elif word in ['mutiny', 'trial', 'arrest', 'betrayal', 'execution']:
                    key_terms.append(word)  # The action itself
            
            # Check if any key terms appear in evidence
            has_footprint = any(term in evidence_text for term in key_terms if len(term) > 2)
            
            if not has_footprint and key_terms:
                silent_events.append(atom)
    
    return silent_events


def classify_claim_type(claim_text: str) -> str:
    """Classify claim type for ensemble routing."""
    claim_lower = claim_text.lower()
    
    # FABRICATED_INVENTION indicators
    fabricated_indicators = [
        'secret', 'hidden', 'conspiracy', 'plot', 'ritual', 'invented',
        'unknown letter', 'diary entry', 'manuscripts', 'concealed'
    ]
    
    if any(indicator in claim_lower for indicator in fabricated_indicators):
        return "FABRICATED_INVENTION"
    
    # SUPPORTED_EVENT indicators
    event_indicators = [
        'met', 'traveled', 'visited', 'witnessed', 'participated',
        'joined', 'escaped', 'fled', 'fought', 'rescued'
    ]
    
    if any(indicator in claim_lower for indicator in event_indicators):
        return "SUPPORTED_EVENT"
    
    # Default to CANONICAL_BACKGROUND
    return "CANONICAL_BACKGROUND"


def compute_salience(claim_text: str) -> str:
    """Compute claim salience."""
    claim_lower = claim_text.lower()
    high_salience_indicators = [
        'secret', 'conspiracy', 'plot', 'betrayal', 'execution',
        'arrest', 'exile', 'revolutionary', 'major', 'important'
    ]
    
    if any(indicator in claim_lower for indicator in high_salience_indicators):
        return "HIGH"
    return "LOW"


def aggregate_final_decision(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], semantic_index: SemanticIndex = None) -> Dict[str, str]:
    """
    Multi-stage evaluation with four novelties:
    1. Event-Gated Contradiction Rule
    2. One-Way Ensemble Authority  
    3. Historical Texture Allowance
    4. UNSUPPORTED Demotion Rule
    """
    
    # Step 1: Check for OVER_SPECIFICATION (more restrictive)
    claim_text = claim.get('claim_text', '').lower()
    
    # Only trigger if multiple specific entities that contradict known timeline/roles
    over_spec_conditions = [
        ('secret society' in claim_text or 'organization' in claim_text),
        ('founded' in claim_text or 'established' in claim_text),
        ('leader' in claim_text or 'head of' in claim_text),
        ('conspiracy' in claim_text and 'plot' in claim_text)  # Multiple conspiracy terms
    ]
    
    # Require multiple conditions for OVER_SPECIFIED
    if sum(over_spec_conditions) >= 2:
        return {
            "final_decision": "contradict",
            "explanation": "OVER_SPECIFIED: Multiple specific organizational details that contradict canon",
            "grounded_verdict": "OVER_SPECIFIED",
            "semantic_verdict": None,
            "method": "OVER_SPECIFIED",
            "atoms_evaluated": 0,
            "violations": 1
        }
    
    # Step 2: Decompose claim into atoms
    print(f"Decomposing claim...")
    atoms = decompose_claim(claim.get('claim_text', ''))
    atoms = atoms[:7]  # Limit to max 7 atoms
    print(f"Got {len(atoms)} atoms")
    
    salience = compute_salience(claim.get('claim_text', ''))
    
    # CANON EXPECTED EVENT RULE - Check for missing timeline anchors
    canon_missing_events = detect_canon_missing_events(atoms, evidence_chunks, salience)
    if canon_missing_events:
        return {
            "final_decision": "contradict",
            "explanation": f"CANON_MISSING: {canon_missing_events[0][:50]}... has zero canon confirmation",
            "grounded_verdict": "CANON_MISSING",
            "semantic_verdict": None,
            "method": "CANON_EXPECTED_EVENT_RULE",
            "atoms_evaluated": len(atoms),
            "violations": len(canon_missing_events)
        }
    
    # NOVELTY 5: Canonical Plausibility Gate - Check for fabricated event anchors
    fabricated_events = detect_fabricated_event_anchors(atoms, evidence_chunks, salience)
    if fabricated_events:
        return {
            "final_decision": "contradict",
            "explanation": f"FABRICATED_EVENT: {fabricated_events[0][:50]}... has no structural footprint in canon",
            "grounded_verdict": "FABRICATED_EVENT",
            "semantic_verdict": None,
            "method": "CANONICAL_PLAUSIBILITY_GATE",
            "atoms_evaluated": len(atoms),
            "violations": len(fabricated_events)
        }
    
    # NOVELTY 3: Historical Texture Allowance - Check before evaluation
    if is_historical_texture_only(atoms):
        return {
            "final_decision": "consistent",
            "explanation": "HISTORICAL_TEXTURE: All atoms are historical texture types",
            "grounded_verdict": "TEXTURE_ALLOWED",
            "semantic_verdict": None,
            "method": "HISTORICAL_TEXTURE",
            "atoms_evaluated": len(atoms),
            "violations": 0
        }
    
    # Step 3: Evaluate each atom independently
    hard_violations = []
    unsupported = []
    no_constraints = []
    supported = []
    event_atom_count = count_event_atoms(atoms)
    unsupported_event_atoms = 0
    
    for i, atom in enumerate(atoms):
        print(f"Evaluating atom {i+1}/{len(atoms)}: {atom[:30]}...")
        atom_claim = {**claim, 'claim_text': atom}
        
        verdict = grounded_constraint_inference(atom_claim, evidence_chunks)
        print(f"Atom {i+1} result: {verdict}")
        
        # Protect emotion/motivation atoms from HARD_VIOLATION
        if verdict == "HARD_VIOLATION" and is_emotion_motivation_atom(atom):
            print(f"Atom {i+1} protected from HARD_VIOLATION (emotion/motivation)")
            verdict = "UNSUPPORTED"
        
        if verdict == "HARD_VIOLATION":
            hard_violations.append(atom)
        elif verdict == "UNSUPPORTED":
            unsupported.append(atom)
            # Track unsupported event atoms for NOVELTY 4
            if classify_atom_semantic_type(atom) == 'EVENT':
                unsupported_event_atoms += 1
        elif verdict == "SUPPORTED":
            supported.append(atom)
        else:
            no_constraints.append(atom)
    
    # Step 4: Apply decision rules with novelties
    
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
    
    # TIERED CANON EVENT SUPPORT RULE
    if salience == "HIGH":
        # Calculate weighted canon violations
        canon_violation_score = 0
        tier3_violations = []
        tier2_violations = []
        
        for atom in unsupported:
            tier = get_canon_event_tier(atom)
            if tier == 3:
                tier3_violations.append(atom)
                canon_violation_score += 3  # Tier 3: 1 failure is enough (score 3)
            elif tier == 2:
                tier2_violations.append(atom)
                canon_violation_score += 1.5  # Tier 2: need 2 failures (score 1.5 each)
            # Tier 1 events never contradict alone (score 0)
        
        # Apply tier-based thresholds
        if tier3_violations:  # Any Tier 3 violation = contradict
            return {
                "final_decision": "contradict",
                "explanation": f"TIER3_VIOLATION: {tier3_violations[0][:50]}... (identity lock unsupported)",
                "grounded_verdict": "TIER3_CANON_VIOLATION",
                "semantic_verdict": None,
                "method": "TIERED_CANON_RULE",
                "atoms_evaluated": len(atoms),
                "violations": len(tier3_violations)
            }
        elif len(tier2_violations) >= 2:  # 2+ Tier 2 violations = contradict
            return {
                "final_decision": "contradict",
                "explanation": f"TIER2_VIOLATIONS: {len(tier2_violations)} timeline anchors unsupported",
                "grounded_verdict": "TIER2_CANON_VIOLATIONS",
                "semantic_verdict": None,
                "method": "TIERED_CANON_RULE",
                "atoms_evaluated": len(atoms),
                "violations": len(tier2_violations)
            }
    
    # Classify claim for routing
    claim_type = classify_claim_type(claim.get('claim_text', ''))
    
    # NOVELTY 4: UNSUPPORTED Demotion Rule - LIGHTER
    unsupported_contributes_to_contradict = (
        unsupported_event_atoms >= 3 and salience == "HIGH"  # Raised from 2 to 3
    )
    
    # Rule 2: Handle UNSUPPORTED with demotion rule - BALANCED
    if unsupported and not unsupported_contributes_to_contradict:
        # Only demote if LOW salience AND no canon-obligated events
        canon_obligated_events = [atom for atom in atoms if is_canon_obligated_event(atom)]
        unsupported_canon_events = [atom for atom in unsupported if is_canon_obligated_event(atom)]
        
        if salience == "LOW" and not canon_obligated_events:
            # Safe to demote - low salience, no important events
            return {
                "final_decision": "consistent",
                "explanation": "UNSUPPORTED_DEMOTED: Low salience, no canon events",
                "grounded_verdict": "UNSUPPORTED",
                "semantic_verdict": None,
                "method": "UNSUPPORTED_DEMOTED",
                "atoms_evaluated": len(atoms),
                "violations": 0
            }
        else:
            # Apply tiered evaluation for unsupported canon events
            tier_score = 0
            for atom in unsupported_canon_events:
                tier = get_canon_event_tier(atom)
                if tier == 3:
                    tier_score += 3
                elif tier == 2:
                    tier_score += 1.5
                # Tier 1 adds 0
            
            if tier_score >= 3:  # Equivalent to 1 Tier 3 or 2 Tier 2 events
                return {
                    "final_decision": "contradict",
                    "explanation": f"TIERED_CANON_UNSUPPORTED: Score {tier_score} from {len(unsupported_canon_events)} events",
                    "grounded_verdict": "UNSUPPORTED",
                    "semantic_verdict": None,
                    "method": "TIERED_CANON_UNSUPPORTED",
                    "atoms_evaluated": len(atoms),
                    "violations": len(unsupported_canon_events)
                }
            else:
                # HIGH salience but no canon events - still demote
                return {
                    "final_decision": "consistent",
                    "explanation": "UNSUPPORTED_DEMOTED: High salience but no canon events",
                    "grounded_verdict": "UNSUPPORTED",
                    "semantic_verdict": None,
                    "method": "UNSUPPORTED_DEMOTED",
                    "atoms_evaluated": len(atoms),
                    "violations": 0
                }
    
    # Base prediction logic
    base_prediction = "consistent"  # Default
    
    if unsupported_contributes_to_contradict:
        base_prediction = "contradict"
    
    # NOVELTY 2: One-Way Ensemble Authority
    ensemble_prediction = None
    ensemble_metadata = None
    
    # Only run ensemble for complex cases, not background-only claims
    if claim_type != "CANONICAL_BACKGROUND" or salience == "HIGH":
        ensemble_prediction, ensemble_metadata = run_ensemble_decision(
            claim, claim_type, salience, evidence_chunks, ensemble_weight=1.0
        )
    
    # Apply one-way ensemble authority
    final_prediction = base_prediction
    method = "BASE_DECISION"
    
    if ensemble_prediction:
        if (base_prediction == "consistent" and 
            ensemble_prediction == "contradict" and
            salience == "HIGH" and 
            event_atom_count > 0):
            # Ensemble may veto CONSISTENT → CONTRADICT only under strict conditions
            final_prediction = "contradict"
            method = "ENSEMBLE_VETO"
        elif base_prediction == "contradict":
            # Ensemble may NOT veto CONTRADICT → CONSISTENT
            final_prediction = base_prediction
            method = "BASE_CONTRADICT_PROTECTED"
    
    # NOVELTY 1: Event-Gated Contradiction Rule (final safety gate)
    if (final_prediction == "contradict" and 
        event_atom_count == 0 and 
        len(hard_violations) == 0):
        # Force override to CONSISTENT
        final_prediction = "consistent"
        method = "BACKGROUND_SAFE_DEFAULT"
    
    return {
        "final_decision": final_prediction,
        "explanation": f"Method: {method}, Events: {event_atom_count}, Violations: {len(hard_violations)}",
        "grounded_verdict": "ENSEMBLE" if ensemble_prediction else "BASE",
        "semantic_verdict": None,
        "method": method,
        "atoms_evaluated": len(atoms),
        "violations": len(hard_violations),
        "event_atom_count": event_atom_count,
        "ensemble_metadata": ensemble_metadata
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