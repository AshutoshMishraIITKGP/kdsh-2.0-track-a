from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()


def run_ensemble_decision(claim: Dict[str, str], claim_type: str, salience: str, retrieved_chunks: List[Dict[str, str]], ensemble_weight: float = 1.0) -> Tuple[str, Dict]:
    """
    Ensemble v1: Three-perspective LLM decision with deterministic aggregation.
    
    Args:
        claim: Claim dictionary with claim_text
        claim_type: FABRICATED_INVENTION, SUPPORTED_EVENT, or CANONICAL_BACKGROUND
        salience: HIGH or LOW
        retrieved_chunks: Evidence chunks
        
    Returns:
        Tuple of (final_label, metadata)
    """
    api_key = os.getenv("GLM_API_KEY", "78e909a9cf7b48a2856a1b178fbd4e7d.ZKmtkKseITStcyrE")
    if not api_key:
        raise RuntimeError("GLM_API_KEY environment variable is required")
    
    client = ZhipuAI(api_key=api_key)
    
    # Format evidence
    evidence_text = format_evidence_chunks(retrieved_chunks)
    claim_text = claim.get('claim_text', '')
    
    # Three perspective prompts
    archivist_prompt = f"""You are acting as THE ARCHIVIST.
Judge whether the claim is CONSISTENT or CONTRADICTORY
with the retrieved evidence.
Only explicit evidence counts. Absence of support = CONTRADICT.
Output ONLY: CONSISTENT or CONTRADICT.

CLAIM: {claim_text}
EVIDENCE: {evidence_text}"""

    historian_prompt = f"""You are acting as THE LITERARY HISTORIAN.
Judge whether the claim fits the character's canon.
Implied traits and narrative continuity are allowed.
Reject only if clearly incompatible with canon.
Output ONLY: CONSISTENT or CONTRADICT.

CLAIM: {claim_text}
EVIDENCE: {evidence_text}"""

    prosecutor_prompt = f"""You are acting as THE PROSECUTOR.
Assume the claim is false until strongly supported.
Escalations and major events require explicit backing.
If any major component lacks support, reject.
Output ONLY: CONSISTENT or CONTRADICT.

CLAIM: {claim_text}
EVIDENCE: {evidence_text}"""
    
    # Get three perspectives
    try:
        archivist_response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": archivist_prompt}],
            temperature=0.1,
            max_tokens=10
        ).choices[0].message.content.strip().upper()
        
        historian_response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": historian_prompt}],
            temperature=0.1,
            max_tokens=10
        ).choices[0].message.content.strip().upper()
        
        prosecutor_response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prosecutor_prompt}],
            temperature=0.1,
            max_tokens=10
        ).choices[0].message.content.strip().upper()
        
        # Normalize responses
        archivist = "CONTRADICT" if "CONTRADICT" in archivist_response else "CONSISTENT"
        historian = "CONTRADICT" if "CONTRADICT" in historian_response else "CONSISTENT"
        prosecutor = "CONTRADICT" if "CONTRADICT" in prosecutor_response else "CONSISTENT"
        
    except Exception as e:
        print(f"Ensemble error: {e}")
        # Fallback to conservative
        archivist = historian = prosecutor = "CONTRADICT"
    
    # Deterministic aggregation logic with background-friendly defaults
    if claim_type == "FABRICATED_INVENTION":
        if archivist == "CONTRADICT":
            final_label = "CONTRADICT"
        else:
            final_label = "CONSISTENT"
    elif claim_type == "SUPPORTED_EVENT":
        if prosecutor == "CONTRADICT":
            final_label = "CONTRADICT"
        elif archivist == "CONSISTENT":
            final_label = "CONSISTENT"
        else:
            final_label = "CONTRADICT"
    elif claim_type == "CANONICAL_BACKGROUND":
        # Change #2: Flip default for CANONICAL_BACKGROUND
        # Change #3: Kill prosecutor veto for background claims
        # Bucket 3 Fix: Apply ensemble weight for event-sparse background claims
        if salience == "LOW":
            # Low-salience background defaults to consistent
            final_label = "CONSISTENT"
        else:
            # High-salience background: historian leads, prosecutor is advisory only
            if historian == "CONSISTENT":
                final_label = "CONSISTENT"
            elif archivist == "CONTRADICT" and ensemble_weight >= 0.8:  # Require high confidence
                final_label = "CONTRADICT"
            else:
                final_label = "CONSISTENT"  # Default to consistent for background
    else:
        # Default fallback
        final_label = "CONTRADICT"
    
    metadata = {
        "decision_source": "ensemble_v1",
        "ensemble_weight": ensemble_weight,
        "perspectives": {
            "archivist": archivist,
            "historian": historian,
            "prosecutor": prosecutor
        }
    }
    
    return final_label.lower(), metadata


def format_evidence_chunks(chunks: List[Dict[str, str]]) -> str:
    """Format evidence chunks for ensemble prompts."""
    if not chunks:
        return "No evidence found."
    
    formatted = []
    for i, chunk in enumerate(chunks[:3], 1):  # Limit to top 3 for prompt length
        formatted.append(f"Evidence {i}: {chunk['text'][:300]}...")
    
    return "\n\n".join(formatted)