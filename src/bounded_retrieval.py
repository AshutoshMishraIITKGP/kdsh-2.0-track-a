"""
Bounded Retrieval Escalation Module
Implements one-time retrieval escalation for high-expectation claims.
"""

from typing import List, Dict


K_BASE = 5  # Initial retrieval count
K_MAX = 10  # Maximum after escalation
DELTA = 5   # Escalation increment


def should_escalate_retrieval(
    claim_classification: Dict[str, any],
    has_hard_violation: bool,
    has_supported: bool
) -> bool:
    """
    Determine if retrieval should be escalated.
    
    Escalate only if:
    - expected_evidence == True
    - no HARD_VIOLATION found
    - no SUPPORTED atoms found
    """
    if not claim_classification.get('expected_evidence', False):
        return False
    
    if has_hard_violation:
        return False
    
    if has_supported:
        return False
    
    return True


def perform_bounded_escalation(
    semantic_index,
    claim: Dict[str, str],
    initial_chunks: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Perform one-time bounded retrieval escalation.
    
    Returns additional chunks (no loops, single escalation only).
    """
    # Retrieve additional chunks up to K_MAX
    additional_chunks = semantic_index.semantic_retrieve(
        claim,
        max_chunks=K_MAX
    )
    
    # Return combined chunks (deduplicated by chunk_id)
    seen_ids = {chunk.get('chunk_id') for chunk in initial_chunks}
    new_chunks = [
        chunk for chunk in additional_chunks
        if chunk.get('chunk_id') not in seen_ids
    ]
    
    return initial_chunks + new_chunks
