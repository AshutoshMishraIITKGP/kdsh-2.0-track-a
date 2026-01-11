from typing import List, Dict
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class GroundedInference:
    """LLM-based grounded constraint inference using Mistral."""
    
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable is required")
        
        self.client = Mistral(api_key=api_key)
        
        self.prompt_template = """You are a Forensic Auditor. Your goal is to find CLEAR LIES.

To avoid False Accusations (False Positives), apply these rules:

1. SILENCE IS NOT CONTRADICTION
   - If the claim mentions a minor detail (birthmark, hobby, sister's name) that is simply NOT MENTIONED in the text, mark it as UNSUPPORTED (NOT HARD_VIOLATION)
   - Texture details are ALLOWED to be absent

2. HARD CONFLICT ONLY
   - Only mark HARD_VIOLATION if you find a "Competing Fact"
   - Example: Claim says 'Tasmania', Text says 'New Zealand' → HARD_VIOLATION
   - Example: Claim says 'Royalist', Text says 'Bonapartist' → HARD_VIOLATION
   - Example: Claim says 'arrested in 1815', Text says 'arrested in 1814' → HARD_VIOLATION

3. PROBABILITY CHECK
   - If claim adds a massive, life-changing event (met Count of Monte Cristo, joined secret society) that is missing from comprehensive bio → UNSUPPORTED
   - Missing major events are UNSUPPORTED, not HARD_VIOLATION

4. COMPETING FACT REQUIREMENT
   - HARD_VIOLATION requires a specific quote that REJECTS the claim
   - If you cannot find text that explicitly contradicts, default to UNSUPPORTED or NO_CONSTRAINT

CATEGORIES:
- SUPPORTED: Text explicitly states the SAME fact with SAME values
- HARD_VIOLATION: Text explicitly states a DIFFERENT value (competing fact exists)
- UNSUPPORTED: Claim adds details not in text (silence, not contradiction)
- NO_CONSTRAINT: Pure opinion/emotion with no factual claim

CRITICAL: Character name + event noun appearing together ≠ SUPPORTED
You need the EXACT relationship stated.

EXAMPLES:

Claim: "Mutiny began when Grant uncovered forged logbook"
Text: "Grant discovered the forged logbook, triggering the mutiny"
Result: SUPPORTED (exact cause stated)

Claim: "Mutiny began when Grant uncovered forged logbook"
Text: "There was a mutiny. Grant was the captain."
Result: UNSUPPORTED (mutiny + Grant mentioned, but cause NOT stated)

Claim: "Mutiny began when Grant uncovered forged logbook"
Text: "The mutiny began due to piracy and abandonment"
Result: HARD_VIOLATION (competing fact: different cause stated)

Claim: "He was in Tasmania"
Text: "The convict escaped in Australia"
Result: UNSUPPORTED (Australia ≠ Tasmania, but not explicit contradiction)

Claim: "He was in Tasmania"
Text: "He fled to New Zealand"
Result: HARD_VIOLATION (competing fact: New Zealand contradicts Tasmania)

Claim: "Noirtier was a Royalist"
Text: "Noirtier was a fervent Bonapartist"
Result: HARD_VIOLATION (competing fact: Bonapartist contradicts Royalist)

Claim: "He had a sister named Marie"
Text: "He grew up in Paris with his family"
Result: UNSUPPORTED (sister not mentioned, but no competing fact)

CLAIM: {claim_text}

TEXT: {evidence_chunks}

Does the text contain a COMPETING FACT that contradicts this claim?
Provide your answer in this format:
VERDICT: [SUPPORTED/HARD_VIOLATION/UNSUPPORTED/NO_CONSTRAINT]
REASON: [Brief explanation of why]"""
    
    def format_evidence(self, evidence_chunks: List[Dict[str, str]]) -> str:
        """Format evidence chunks for the prompt."""
        if not evidence_chunks:
            return "No evidence chunks found."
        
        formatted = []
        for i, chunk in enumerate(evidence_chunks, 1):
            formatted.append(f"Chunk {i}:\n{chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}")
        
        return "\n\n".join(formatted)
    
    def infer_grounded_constraint(self, claim: Dict[str, str], evidence_chunks: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Perform grounded constraint inference using Mistral.
        
        Returns:
            Dict with 'verdict' and 'reason'
        """
        try:
            evidence_text = self.format_evidence(evidence_chunks)
            
            prompt_text = self.prompt_template.format(
                claim_text=claim.get('claim_text', ''),
                evidence_chunks=evidence_text
            )
            
            response = self.client.chat.complete(
                model="mistral-small-2503",
                messages=[{"role": "user", "content": prompt_text}]
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse verdict and reason
            verdict = "UNSUPPORTED"
            reason = "No clear evidence found"
            
            if "VERDICT:" in response_text:
                lines = response_text.split('\n')
                for line in lines:
                    if line.startswith("VERDICT:"):
                        verdict = line.replace("VERDICT:", "").strip().upper()
                    elif line.startswith("REASON:"):
                        reason = line.replace("REASON:", "").strip()
            else:
                # Fallback to old format
                response_upper = response_text.upper()
                if "HARD_VIOLATION" in response_upper:
                    verdict = "HARD_VIOLATION"
                elif "SUPPORTED" in response_upper:
                    verdict = "SUPPORTED"
                elif "UNSUPPORTED" in response_upper:
                    verdict = "UNSUPPORTED"
                else:
                    verdict = "NO_CONSTRAINT"
                reason = response_text[:200]
            
            # Validate SUPPORTED verdicts
            if "SUPPORTED" in verdict:
                claim_text_lower = claim.get('claim_text', '').lower()
                evidence_text_lower = evidence_text.lower()
                claim_words = set(claim_text_lower.split())
                common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'was', 'were', 'is', 'are', 'he', 'she', 'it', 'they'}
                key_words = claim_words - common_words
                matches = sum(1 for word in key_words if word in evidence_text_lower)
                if len(key_words) > 0 and matches / len(key_words) < 0.5:
                    verdict = "UNSUPPORTED"
                    reason = "Insufficient word overlap in evidence"
            
            return {"verdict": verdict, "reason": reason}
                
        except Exception as e:
            print(f"Grounded inference error: {e}")
            return {"verdict": "UNSUPPORTED", "reason": str(e)}


def grounded_constraint_inference(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Standalone function for grounded constraint inference.
    
    Returns:
        Dict with 'verdict' and 'reason'
    """
    if not evidence_chunks:
        return {"verdict": "UNSUPPORTED", "reason": "No evidence chunks provided"}
    
    inference = GroundedInference()
    return inference.infer_grounded_constraint(claim, evidence_chunks)