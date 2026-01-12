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
        
        self.prompt_template = """You are a Forensic Auditor. Your goal is to find LIES and CONTRADICTIONS.

STRICT RULES:

1. COMPETING FACT = HARD_VIOLATION
   - If text explicitly states a DIFFERENT value, mark HARD_VIOLATION
   - Example: Claim says 'Tasmania', Text says 'New Zealand' -> HARD_VIOLATION
   - Example: Claim says 'Royalist', Text says 'Bonapartist' -> HARD_VIOLATION
   - Example: Claim says 'arrested in 1815', Text says 'arrested in 1814' -> HARD_VIOLATION

2. UNDERSTAND PARAPHRASE AND GRAMMAR VARIATIONS
   - Same meaning with different words = SUPPORTED
   - "arrested" = "taken into custody" = "imprisoned" = "detained"
   - "met" = "encountered" = "came across" = "was introduced to"
   - "fled to" = "escaped to" = "ran away to" = "sought refuge in"
   - Active vs passive voice is the SAME fact: "He arrested X" = "X was arrested by him"
   - Past vs present tense describing same event is the SAME fact
   - Different sentence structure but same meaning = SUPPORTED

3. GEOGRAPHIC/POLITICAL CONTRADICTIONS
   - Different locations ARE contradictions (Tasmania vs New Zealand)
   - BUT: General region vs specific location is NOT contradiction
     - "Australia" and "Tasmania" can coexist (Tasmania is in Australia)
     - "Europe" and "France" can coexist (France is in Europe)
   - Different political affiliations ARE contradictions (Royalist vs Bonapartist)
   - Be aggressive on factual contradictions

4. SILENCE IS UNSUPPORTED
   - If detail is NOT MENTIONED, mark UNSUPPORTED
   - Major events missing = UNSUPPORTED (not HARD_VIOLATION)

5. SUPPORTED REQUIRES SIMILAR MEANING MATCH
   - Text must state the same MEANING (not same words)
   - Accept paraphrase, synonyms, grammar changes
   - "He was arrested in Paris" = "Paris authorities detained him" = SUPPORTED

CATEGORIES:
- SUPPORTED: Text confirms the same MEANING (accept paraphrase/grammar changes)
- HARD_VIOLATION: Text states a DIFFERENT value (be aggressive)
- UNSUPPORTED: Claim adds details not in text
- NO_CONSTRAINT: Pure opinion/emotion

EXAMPLES:

Claim: "Mutiny began when Grant uncovered forged logbook"
Text: "Grant's discovery of the forged logbook triggered the mutiny"
Result: SUPPORTED (same meaning, different grammar)

Claim: "He was arrested in Paris"
Text: "Paris authorities detained him"
Result: SUPPORTED (arrested = detained)

Claim: "He was in Tasmania"
Text: "The convict escaped in Australia"
Result: SUPPORTED (Tasmania is in Australia)

Claim: "He was in Tasmania"
Text: "He fled to New Zealand"
Result: HARD_VIOLATION (New Zealand contradicts Tasmania)

Claim: "Noirtier was a Royalist"
Text: "Noirtier was a fervent Bonapartist"
Result: HARD_VIOLATION (Bonapartist contradicts Royalist)

CLAIM: {claim_text}

TEXT: {evidence_chunks}

Does the text contain a COMPETING FACT? Accept paraphrase and grammar variations.
VERDICT: [SUPPORTED/HARD_VIOLATION/UNSUPPORTED/NO_CONSTRAINT]
REASON: [Brief explanation]"""
    
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
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
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
