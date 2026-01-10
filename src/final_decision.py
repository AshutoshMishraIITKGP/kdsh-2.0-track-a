from typing import List, Dict
import time
from claim_decomposer import decompose_claim
from grounded_inference import grounded_constraint_inference
from semantic_neighborhood import semantic_neighborhood_evaluation
from semantic_index import SemanticIndex
from ensemble_v1 import run_ensemble_decision
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()


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
    
    return False


def is_unique_interaction_event(atom: str) -> bool:
    """Check if atom describes a unique interaction that would anchor timelines."""
    atom_lower = atom.lower()
    
    unique_interaction_indicators = [
        # Meetings and encounters
        'met', 'meeting', 'encountered', 'spoke with', 'confronted',
        # Arrests and captures
        'arrested', 'captured', 'seized', 'detained', 'imprisoned',
        # Betrayals and alliances
        'betrayed', 'allied', 'joined forces', 'conspired',
        # Deaths and violence
        'killed', 'murdered', 'executed', 'assassinated',
        # Rescues and escapes
        'rescued', 'saved', 'freed', 'escaped with'
    ]
    
    return any(indicator in atom_lower for indicator in unique_interaction_indicators)


def is_central_event(atom: str) -> bool:
    """Check if atom represents a CENTRAL_CANON_EVENT that explains, alters, or precedes known canon anchors."""
    atom_lower = atom.lower()
    
    # Hard blacklist - NEVER trigger on these categories
    identity_blacklist = [
        'birth', 'born', 'birthplace', 'birthmark', 'flame-shaped',
        'childhood', 'family', 'clan', 'tribe', 'heritage', 'ancestry',
        'physical', 'appearance', 'mark', 'scar', 'tattoo',
        'folklore', 'legend', 'tradition', 'ritual', 'ceremony',
        'belief', 'faith', 'religion', 'spiritual', 'cultural',
        'background', 'origin', 'early', 'young', 'youth'
    ]
    
    # If any blacklist term present, never trigger
    if any(term in atom_lower for term in identity_blacklist):
        return False
    
    # Canon anchor events that must be in canon
    canon_anchor_indicators = [
        'arrested', 'imprisonment', 'executed', 'killed', 'murdered',
        'mutiny', 'betrayed', 'betrayal', 'conspiracy', 'plot',
        'first met', 'initially met', 'founded', 'established',
        'political reversal', 'exile', 'banished', 'war', 'battle'
    ]
    
    # Check if atom explains/causes/initiates a canon anchor
    explains_canon_anchor = any(indicator in atom_lower for indicator in canon_anchor_indicators)
    
    # Check for state change language (alive→dead, free→imprisoned, loyal→traitor)
    state_change_indicators = [
        'became', 'turned into', 'transformed', 'changed from', 'shifted to',
        'imprisoned', 'freed', 'promoted', 'dismissed', 'appointed'
    ]
    
    changes_canon_state = any(indicator in atom_lower for indicator in state_change_indicators)
    
    return explains_canon_anchor or changes_canon_state


def is_causal_anchor(atom: str) -> bool:
    """Check if atom is a CAUSAL_ANCHOR that explains why a known canon event occurred."""
    atom_lower = atom.lower()
    
    # Check for causal language
    causal_indicators = [
        'because', 'led to', 'triggered', 'began when', 'resulted in',
        'caused', 'due to', 'as a result of', 'which led to', 'causing'
    ]
    
    has_causal_language = any(indicator in atom_lower for indicator in causal_indicators)
    if not has_causal_language:
        return False
    
    # Check for known canon events mentioned
    canon_events = [
        'mutiny', 'arrest', 'betrayal', 'escape', 'execution', 'death',
        'imprisonment', 'exile', 'war', 'battle', 'revolution', 'conspiracy',
        'plot', 'alliance', 'marriage', 'divorce', 'promotion', 'dismissal'
    ]
    
    has_canon_event = any(event in atom_lower for event in canon_events)
    
    return has_causal_language and has_canon_event


def is_identity_or_background(atom: str) -> bool:
    """Check if atom is identity/background that should never trigger CENTRAL_EVENT_UNSUPPORTED."""
    atom_lower = atom.lower()
    
    identity_background_indicators = [
        'birth', 'born', 'birthplace', 'birthmark', 'childhood', 'family',
        'romance', 'romantic', 'love', 'attachment', 'care', 'caring',
        'belief', 'believed', 'faith', 'physical', 'trait', 'appearance',
        'emotional', 'trauma', 'inspiration', 'inspired', 'personal',
        'background', 'heritage', 'ancestry', 'origin', 'early'
    ]
    
    return any(indicator in atom_lower for indicator in identity_background_indicators)


def involves_two_named_characters(atom: str) -> bool:
    """Check if atom involves two named characters."""
    words = atom.split()
    named_chars = []
    
    for word in words:
        if (word[0].isupper() and len(word) > 2 and word.isalpha() and 
            word not in ['The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'With', 'By']):
            named_chars.append(word.lower())
    
    # Also check for hyphenated names like Kai-Koumou
    for word in words:
        if '-' in word and word[0].isupper():
            named_chars.append(word.lower())
    
    return len(named_chars) >= 2


def is_canon_meeting(atom: str) -> bool:
    """Check if atom asserts a meeting between named canon characters."""
    atom_lower = atom.lower()
    
    # Meeting indicators
    meeting_indicators = ['met', 'meeting', 'encountered', 'first met', 'initially met', 'watched', 'recognition']
    has_meeting = any(indicator in atom_lower for indicator in meeting_indicators)
    
    return has_meeting and involves_two_named_characters(atom)


def no_canon_confirmation(atom: str, evidence_chunks: List[Dict[str, str]]) -> bool:
    """Check if atom has no canon confirmation in evidence."""
    return not has_canon_confirmation(atom, evidence_chunks)


def extract_character_spatiotemporal_info(evidence_chunks: List[Dict[str, str]]) -> Dict[str, Dict]:
    """Extract basic location/time info for characters from evidence."""
    character_info = {}
    
    for chunk in evidence_chunks:
        text = chunk.get('text', '').lower()
        
        # Extract character mentions with location/time context
        characters = ['faria', 'villefort', 'ayrton', 'kai-koumou', 'thalcave', 'noirtier', 'paganel']
        
        for char in characters:
            if char in text:
                if char not in character_info:
                    character_info[char] = {'locations': set(), 'time_periods': set(), 'status': set()}
                
                # Location indicators
                if any(loc in text for loc in ['prison', 'château d\'if', 'marseilles']):
                    character_info[char]['locations'].add('prison')
                if any(loc in text for loc in ['paris', 'france']):
                    character_info[char]['locations'].add('france')
                if any(loc in text for loc in ['ship', 'sea', 'voyage']):
                    character_info[char]['locations'].add('sea')
                if any(loc in text for loc in ['new zealand', 'island']):
                    character_info[char]['locations'].add('new_zealand')
                
                # Time period indicators
                if any(period in text for period in ['revolution', '1793', '1794']):
                    character_info[char]['time_periods'].add('revolution')
                if any(period in text for period in ['1815', 'napoleon', 'hundred days']):
                    character_info[char]['time_periods'].add('1815')
                if any(period in text for period in ['1838', '1839', '1840']):
                    character_info[char]['time_periods'].add('1838-1840')
                
                # Status indicators
                if any(status in text for status in ['imprisoned', 'prison', 'captive']):
                    character_info[char]['status'].add('imprisoned')
                if any(status in text for status in ['dead', 'died', 'death']):
                    character_info[char]['status'].add('dead')
                if any(status in text for status in ['exile', 'exiled']):
                    character_info[char]['status'].add('exiled')
    
    return character_info

def check_character_copresence_possibility(atom: str, evidence_chunks: List[Dict[str, str]]) -> bool:
    """Check if claimed character interaction is spatiotemporally possible."""
    atom_lower = atom.lower()
    
    # Extract character names from atom
    characters = ['faria', 'villefort', 'ayrton', 'kai-koumou', 'thalcave', 'noirtier', 'paganel']
    mentioned_chars = [char for char in characters if char in atom_lower]
    
    if len(mentioned_chars) < 2:
        return True  # Not a multi-character interaction
    
    # Get spatiotemporal info
    char_info = extract_character_spatiotemporal_info(evidence_chunks)
    
    # Check for obvious impossibilities
    for i, char1 in enumerate(mentioned_chars):
        for char2 in mentioned_chars[i+1:]:
            if char1 in char_info and char2 in char_info:
                info1 = char_info[char1]
                info2 = char_info[char2]
                
                # Check for temporal impossibility
                if ('dead' in info1['status'] and len(info2['time_periods']) > 0) or \
                   ('dead' in info2['status'] and len(info1['time_periods']) > 0):
                    # One character dead during other's active period
                    return False
                
                # Check for spatial impossibility
                if info1['locations'] and info2['locations']:
                    # If both have known locations and they never overlap
                    if not info1['locations'].intersection(info2['locations']):
                        # Check for prison constraint
                        if 'prison' in info1['locations'] and 'prison' not in info2['locations']:
                            return False
                        if 'prison' in info2['locations'] and 'prison' not in info1['locations']:
                            return False
    
    return True  # No obvious impossibility detected

def compute_canon_density(character: str, evidence_chunks: List[Dict[str, str]]) -> float:
    """Compute canon density for character (high = well-documented, low = sparse)."""
    char_lower = character.lower()
    
    # Count chunks mentioning character
    mentions = 0
    total_char_text = 0
    
    for chunk in evidence_chunks:
        text = chunk.get('text', '').lower()
        if char_lower in text:
            mentions += 1
            total_char_text += len(text)
    
    if mentions == 0:
        return 0.0
    
    # Density = mentions per chunk + average text length factor
    density = mentions + (total_char_text / mentions) / 1000  # Normalize text length
    
    # Known high-density characters get boost
    high_density_chars = ['faria', 'villefort', 'noirtier', 'dantes', 'monte cristo']
    if any(hdc in char_lower for hdc in high_density_chars):
        density *= 1.5
    
    return density

def strengthen_canon_expected_interactions(atoms: List[str], evidence_chunks: List[Dict[str, str]], salience: str) -> List[str]:
    """Strengthened Canon-Expected Interaction Rule for explicit character interactions."""
    if salience != "HIGH":
        return []
    
    violations = []
    evidence_text = " ".join([chunk.get('text', '') for chunk in evidence_chunks]).lower()
    
    for atom in atoms:
        atom_lower = atom.lower()
        
        # Check for explicit interactions
        interaction_verbs = ['met', 'recruited', 'fought', 'helped', 'worked with', 'allied with', 'conspired']
        has_interaction = any(verb in atom_lower for verb in interaction_verbs)
        
        if not has_interaction:
            continue
        
        # Extract character names
        characters = ['faria', 'villefort', 'ayrton', 'kai-koumou', 'thalcave', 'noirtier', 'paganel']
        mentioned_chars = [char for char in characters if char in atom_lower]
        
        if len(mentioned_chars) < 2:
            continue
        
        # Check for zero co-mention in canon
        has_co_mention = False
        for chunk in evidence_chunks:
            chunk_text = chunk.get('text', '').lower()
            chars_in_chunk = [char for char in mentioned_chars if char in chunk_text]
            if len(chars_in_chunk) >= 2:
                has_co_mention = True
                break
        
        # If no co-mention found, this is a violation
        if not has_co_mention:
            violations.append(atom)
    
def detect_copresence_violations(atoms: List[str], evidence_chunks: List[Dict[str, str]], salience: str) -> List[str]:
    """Detect character interactions that are spatiotemporally impossible."""
    if salience != "HIGH":
        return []
    
    violations = []
    
    for atom in atoms:
        # Only check interaction events
        if not is_unique_interaction_event(atom):
            continue
        
        # Only check multi-character events
        if not is_named_character_event(atom):
            continue
        
        # Check if interaction is spatiotemporally possible
        if not check_character_copresence_possibility(atom, evidence_chunks):
            violations.append(atom)
    
def detect_canon_expected_events(atoms: List[str], evidence_chunks: List[Dict[str, str]], salience: str) -> List[str]:
    """CANON_EXPECTED_EVENT Rule: Detect canon anchor events that must exist in timeline."""
    if salience != "HIGH":
        return []
    
    canon_expected_violations = []
    
    for atom in atoms:
        # Must be EVENT type
        if classify_atom_semantic_type(atom) != 'EVENT':
            continue
        
        # Must be a canon anchor event (not enrichment)
        atom_lower = atom.lower()
        canon_anchor_events = [
            'arrested', 'arrest', 'imprisonment', 'executed', 'killed',
            'mutiny', 'betrayed', 'betrayal', 'conspiracy', 'plot',
            'first met', 'initially met', 'war', 'battle', 'exile'
        ]
        
        is_canon_anchor = any(anchor in atom_lower for anchor in canon_anchor_events)
        if not is_canon_anchor:
            continue
        
        # Must involve named characters
        if not is_named_character_event(atom):
            continue
        
        # Must have zero canon confirmation
        if has_canon_confirmation(atom, evidence_chunks):
            continue
        
        # All conditions met - this is a CANON_EXPECTED violation
        canon_expected_violations.append(atom)
    
    return canon_expected_violations


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


def last_chance_plausibility_veto(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], current_decision: str) -> Dict[str, str]:
    """LAST_CHANCE_PLAUSIBILITY_VETO - Only flips CONSISTENT → CONTRADICT."""
    
    if current_decision != "consistent":
        return {"decision": current_decision, "method": "NO_VETO_NEEDED"}
    
    try:
        api_key = os.getenv("GLM_API_KEY")
        if not api_key:
            return {"decision": current_decision, "method": "NO_API_KEY"}
        
        client = ZhipuAI(api_key=api_key)
        
        # Format evidence
        evidence_text = "\n\n".join([f"Chunk {i+1}: {chunk.get('text', '')[:300]}..." 
                                    for i, chunk in enumerate(evidence_chunks[:3])])
        
        prompt = f"""You are performing a LAST-CHANCE PLAUSIBILITY CHECK for a narrative consistency system.

Your role is NOT to verify evidence.
Your role is NOT to require documentation.
Your role is NOT to be strict.

Assume:
- Missing details are allowed.
- Background events may be unstated in canon.
- Emotional, symbolic, or private experiences are allowed.

Your ONLY task:
Determine whether the claim FORCES a contradiction with established canon.

Return "IMPLAUSIBLE" ONLY if at least one of the following is true:
- The claim invents a specific cause for a known canon event without support.
- The claim introduces a first-time meeting or relationship that canon would have mentioned.
- The claim alters known timelines, deaths, imprisonments, or political roles.
- The claim requires two characters to interact in a way canon explicitly forbids or contradicts.

If the claim could coexist with canon even hypothetically, return "PLAUSIBLE".

CLAIM: {claim.get('claim_text', '')}
CHARACTER: {claim.get('character', '')}
EVIDENCE: {evidence_text}

Output format:
PLAUSIBLE
or
IMPLAUSIBLE: <one-sentence reason>"""
        
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        verdict = response.choices[0].message.content.strip()
        
        if verdict.startswith("IMPLAUSIBLE"):
            reason = verdict.split(":", 1)[1].strip() if ":" in verdict else "Forces canon contradiction"
            return {
                "decision": "contradict",
                "method": "PLAUSIBILITY_VETO",
                "reason": reason
            }
        
        return {"decision": current_decision, "method": "PLAUSIBLE"}
        
    except Exception as e:
        # On error, don't veto
        return {"decision": current_decision, "method": "VETO_ERROR"}


def is_event_claim(atoms: List[str]) -> bool:
    """Detect if claim is an EVENT CLAIM based on atoms."""
    
    for atom in atoms:
        atom_lower = atom.lower()
        
        # Verbs of occurrence
        occurrence_verbs = [
            'met', 'arrested', 'joined', 'escaped', 'hid', 'led', 'burned', 
            'poisoned', 'married', 'rescued', 'betrayed', 'organized', 
            'founded', 'established', 'executed', 'killed', 'fought'
        ]
        
        # Temporal anchors
        temporal_indicators = [
            'in 1815', 'during', 'after', 'before', 'at eighteen', 
            'when', 'while', 'until', 'since'
        ]
        
        # Role changes
        role_changes = [
            'became head', 'adopted alias', 'joined society', 'became',
            'turned into', 'appointed', 'promoted', 'dismissed'
        ]
        
        if (any(verb in atom_lower for verb in occurrence_verbs) or
            any(temporal in atom_lower for temporal in temporal_indicators) or
            any(role in atom_lower for role in role_changes)):
            return True
    
    return False


def is_canon_noticeable_event(claim_text: str) -> bool:
    """Check if event is something canon would have mentioned."""
    claim_lower = claim_text.lower()
    
    canon_noticeable_events = [
        'arrest', 'escape', 'mutiny', 'betray', 'join society',
        'lead', 'organize', 'execute', 'hide', 'marry',
        'secret society', 'conspiracy', 'plot',
        'burn village', 'poison', 'murder', 'rescue'
    ]
    
    # Special case for meetings with major characters
    major_characters = ['monte cristo', 'count', 'fernand', 'villefort', 'noirtier']
    if 'met' in claim_lower:
        if any(char in claim_lower for char in major_characters):
            return True
    
    return any(event in claim_lower for event in canon_noticeable_events)


def requires_evidence_obligation(claim_text: str) -> bool:
    """Check if claim is EVIDENCE-OBLIGATED and cannot use default consistent."""
    claim_lower = claim_text.lower()
    
    # First-time interaction between named characters
    first_time_indicators = ['met', 'first met', 'encountered', 'initially met']
    has_first_time = any(indicator in claim_lower for indicator in first_time_indicators)
    
    # Public action (trial, arrest, leadership, meeting, mutiny cause)
    public_action_indicators = [
        'trial', 'arrest', 'arrested', 'leadership', 'led', 'commanded',
        'meeting', 'mutiny', 'caused', 'triggered', 'public', 'announced'
    ]
    has_public_action = any(indicator in claim_lower for indicator in public_action_indicators)
    
    # Causal explanation for known canon event
    causal_indicators = ['because', 'led to', 'triggered', 'caused', 'resulted in']
    canon_events = ['mutiny', 'arrest', 'betrayal', 'escape', 'execution', 'war', 'battle']
    has_causal_canon = (any(causal in claim_lower for causal in causal_indicators) and 
                       any(event in claim_lower for event in canon_events))
    
    # Institutional affiliation or secret society
    institutional_indicators = [
        'member of', 'joined', 'society', 'organization', 'order', 'league',
        'brotherhood', 'affiliated', 'secret', 'conspiracy'
    ]
    has_institutional = any(indicator in claim_lower for indicator in institutional_indicators)
    
    # Political maneuvering with real consequences
    political_indicators = [
        'political', 'politics', 'government', 'revolution', 'royalist',
        'bonapartist', 'conspiracy', 'plot', 'alliance', 'betrayal'
    ]
    has_political = any(indicator in claim_lower for indicator in political_indicators)
    
    return (has_first_time or has_public_action or has_causal_canon or 
            has_institutional or has_political)


def run_obligation_ensemble(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]]) -> str:
    """Run ensemble for NEEDS_JUSTIFICATION cases with new question."""
    try:
        api_key = os.getenv("GLM_API_KEY")
        if not api_key:
            return "PLAUSIBLE_UNDOCUMENTED"
        
        client = ZhipuAI(api_key=api_key)
        
        # Format evidence
        evidence_text = "\n\n".join([f"Chunk {i+1}: {chunk.get('text', '')[:300]}..." 
                                    for i, chunk in enumerate(evidence_chunks[:3])])
        
        prompt = f"""You are evaluating whether a claim would reasonably be documented or implied in canon if it were true.

Your task: Determine if this type of claim would leave traces in the narrative.

Consider:
- Would this event intersect with documented storylines?
- Would this relationship affect known character arcs?
- Would this action have consequences that canon would mention?

Return "IMPLAUSIBLE_CANON_BREAK" if:
- The claim describes events that would necessarily intersect with documented canon
- The relationships or actions would have affected known storylines
- Canon would have mentioned this if it were true

Return "PLAUSIBLE_UNDOCUMENTED" if:
- The claim could occur "between the lines" of canon
- It doesn't contradict or require changes to known events
- It's the type of detail canon might not document

CLAIM: {claim.get('claim_text', '')}
CHARACTER: {claim.get('character', '')}
EVIDENCE: {evidence_text}

Output format:
PLAUSIBLE_UNDOCUMENTED
or
IMPLAUSIBLE_CANON_BREAK: <one-sentence explanation>"""
        
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        
        verdict = response.choices[0].message.content.strip()
        return verdict
        
    except Exception as e:
        # On error, default to plausible
        return "PLAUSIBLE_UNDOCUMENTED"


def has_structural_event(atoms: List[str]) -> bool:
    """Check if claim has structural event that requires canon support."""
    
    STRUCTURAL_VERBS = [
        "arrest", "imprison", "escape", "mutiny",
        "betray", "marry", "execute", "die",
        "join society", "become leader", "form organization",
        "first meet", "reveal identity", "revealed"
    ]
    
    for atom in atoms:
        atom_lower = atom.lower()
        
        # Check for structural verbs
        for verb in STRUCTURAL_VERBS:
            if verb in atom_lower:
                return True
        
        # Check for two major named characters in first-time interaction
        major_characters = ['monte cristo', 'count', 'fernand', 'villefort', 'noirtier', 'faria']
        first_time_indicators = ['met', 'first met', 'encountered', 'initially met']
        
        if any(indicator in atom_lower for indicator in first_time_indicators):
            char_count = sum(1 for char in major_characters if char in atom_lower)
            if char_count >= 2:
                return True
        
        # Check for legal/social status changes
        status_changes = [
            'became', 'appointed', 'promoted', 'dismissed', 'joined',
            'member of', 'leader of', 'head of'
        ]
        if any(change in atom_lower for change in status_changes):
            return True
    
    return False


def has_unsupported_atoms(atoms: List[str], evidence_chunks: List[Dict[str, str]]) -> bool:
    """Check if claim has unsupported atoms."""
    for atom in atoms:
        atom_claim = {'claim_text': atom}
        verdict = grounded_constraint_inference(atom_claim, evidence_chunks)
        if verdict == "UNSUPPORTED":
            return True
    return False


CLOSED_EVENT_COUNT = {
    "arrest", "imprisonment", "execution", "death", "escape"
}

CLOSED_RELATIONAL_FIRSTS = {
    "first_meeting", "initial_contact", "betrayal_between_characters"
}

OPEN_WORLD_DESCRIPTIVE = {
    "birthmark", "habit", "ritual", "symbolism", "personal belief", "emotional reaction"
}

CAUSAL_LANGUAGE = [
    "began when", "because", "led to", "triggered", "caused", "resulted in"
]


def introduces_additional_instance(claim_text: str) -> bool:
    """Check if claim introduces additional instance of closed event."""
    claim_lower = claim_text.lower()
    return any(event in claim_lower for event in CLOSED_EVENT_COUNT)


def introduces_first_contact_between_known_characters(claim_text: str) -> bool:
    """Check if claim introduces first contact between known characters."""
    claim_lower = claim_text.lower()
    
    first_contact_indicators = ["first met", "initially met", "first encountered", "initial contact"]
    major_characters = ['monte cristo', 'count', 'fernand', 'villefort', 'noirtier', 'faria']
    
    has_first_contact = any(indicator in claim_lower for indicator in first_contact_indicators)
    char_count = sum(1 for char in major_characters if char in claim_lower)
    
    return has_first_contact and char_count >= 2


def contains_causal_language_and_canon_event(claim_text: str) -> bool:
    """Check if claim contains causal language and references known canon event."""
    claim_lower = claim_text.lower()
    
    has_causal = any(causal in claim_lower for causal in CAUSAL_LANGUAGE)
    known_canon_events = ['political', 'betrayal', 'revenge', 'conspiracy', 'drift', 'change']
    references_canon = any(event in claim_lower for event in known_canon_events)
    
    return has_causal and references_canon


def is_open_world_descriptive(claim_text: str) -> bool:
    """Check if claim is open-world descriptive type."""
    claim_lower = claim_text.lower()
    return any(desc_type in claim_lower for desc_type in OPEN_WORLD_DESCRIPTIVE)


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
    """Three-type CLOSED_CANON decision logic."""
    
    claim_text = claim.get('claim_text', '')
    
    # Decompose claim into atoms
    atoms = decompose_claim(claim_text)
    atoms = atoms[:7]
    
    # Check if unsupported
    unsupported = has_unsupported_atoms(atoms, evidence_chunks)
    
    # OPEN_WORLD_DESCRIPTIVE bypass (never trigger closed canon)
    if is_open_world_descriptive(claim_text):
        return {
            "final_decision": "consistent",
            "explanation": "OPEN_WORLD_DESCRIPTIVE: Descriptive claim allowed in open world",
            "grounded_verdict": "OPEN_WORLD_DESCRIPTIVE",
            "semantic_verdict": None,
            "method": "OPEN_WORLD_DESCRIPTIVE",
            "atoms_evaluated": len(atoms),
            "violations": 0
        }
    
    # A. CLOSED_EVENT_COUNT (hard)
    if introduces_additional_instance(claim_text):
        if unsupported:
            return {
                "final_decision": "contradict",
                "explanation": "CLOSED_EVENT_COUNT: Introduces additional instance of closed event",
                "grounded_verdict": "CLOSED_EVENT_COUNT",
                "semantic_verdict": None,
                "method": "CLOSED_EVENT_COUNT",
                "atoms_evaluated": len(atoms),
                "violations": 1
            }
    
    # B. CLOSED_RELATIONAL_FIRSTS (hard)
    if introduces_first_contact_between_known_characters(claim_text):
        if unsupported:
            return {
                "final_decision": "contradict",
                "explanation": "CLOSED_RELATIONAL_FIRSTS: Introduces unsupported first contact between known characters",
                "grounded_verdict": "CLOSED_RELATIONAL_FIRSTS",
                "semantic_verdict": None,
                "method": "CLOSED_RELATIONAL_FIRSTS",
                "atoms_evaluated": len(atoms),
                "violations": 1
            }
    
    # C. CAUSAL_OVERWRITE (hard, single line)
    if contains_causal_language_and_canon_event(claim_text):
        if unsupported:
            return {
                "final_decision": "contradict",
                "explanation": "CAUSAL_OVERWRITE: Unsupported causal explanation for known canon event",
                "grounded_verdict": "CAUSAL_OVERWRITE",
                "semantic_verdict": None,
                "method": "CAUSAL_OVERWRITE",
                "atoms_evaluated": len(atoms),
                "violations": 1
            }
    
    # Default logic (correct and should stay)
    return {
        "final_decision": "consistent",
        "explanation": "No closed canon violations detected",
        "grounded_verdict": "CONSISTENT",
        "semantic_verdict": None,
        "method": "DEFAULT_CONSISTENT",
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