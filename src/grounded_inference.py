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
        
        # STRICT PROMPT: Exact matching, minimal semantic flexibility
        self.strict_prompt = """You are a Strict Fact Checker. Find explicit contradictions with minimal interpretation.

RULES:
1. COMPETING FACT = HARD_VIOLATION (different values stated)
2. Accept ONLY close paraphrases (arrested = detained, met = encountered)
3. Different locations = HARD_VIOLATION (Tasmania vs New Zealand)
4. Missing details = UNSUPPORTED
5. Require explicit textual match for SUPPORTED

CATEGORIES:
- SUPPORTED: Text explicitly confirms (minimal paraphrase accepted)
- HARD_VIOLATION: Text states different value
- UNSUPPORTED: Not mentioned
- NO_CONSTRAINT: Opinion/emotion

CLAIM: {claim_text}
TEXT: {evidence_chunks}

VERDICT: [SUPPORTED/HARD_VIOLATION/UNSUPPORTED/NO_CONSTRAINT]
REASON: [Brief explanation]"""
        
        # MODERATE PROMPT: Balanced semantic understanding
        self.moderate_prompt = """You are a Forensic Auditor. Find contradictions while accepting semantic equivalence.

RULES:
1. COMPETING FACT = HARD_VIOLATION
2. Accept paraphrase and grammar variations (arrested = detained = imprisoned)
3. Active/passive voice = same fact
4. Geographic hierarchy OK (Tasmania in Australia)
5. Missing details = UNSUPPORTED
6. SUPPORTED requires semantic match

CATEGORIES:
- SUPPORTED: Same meaning (accept paraphrase/grammar)
- HARD_VIOLATION: Different value stated
- UNSUPPORTED: Not mentioned
- NO_CONSTRAINT: Opinion/emotion

CLAIM: {claim_text}
TEXT: {evidence_chunks}

VERDICT: [SUPPORTED/HARD_VIOLATION/UNSUPPORTED/NO_CONSTRAINT]
REASON: [Brief explanation]"""
        
        # LENIENT PROMPT: Maximum semantic flexibility
        self.lenient_prompt = """You are a Narrative Analyst. Evaluate claims with semantic understanding and contextual reasoning.

RULES:
1. COMPETING FACT = HARD_VIOLATION (only clear contradictions)
2. Accept broad semantic equivalence (arrested = detained = imprisoned = taken into custody)
3. Accept synonyms, paraphrases, grammar variations, tense changes
4. Accept implicit meanings and reasonable inferences
5. Geographic/temporal flexibility (Tasmania in Australia, nearby times)
6. Missing details = UNSUPPORTED (not violations)
7. SUPPORTED if meaning aligns

CATEGORIES:
- SUPPORTED: Meaning aligns (broad semantic match)
- HARD_VIOLATION: Clear contradiction only
- UNSUPPORTED: Not mentioned
- NO_CONSTRAINT: Opinion/emotion

CLAIM: {claim_text}
TEXT: {evidence_chunks}

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
    
    def infer_grounded_constraint(self, claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], perspective: str = "moderate") -> Dict[str, str]:
        """
        Perform grounded constraint inference using Mistral.
        
        Args:
            perspective: "strict" (temp=0.0), "moderate" (temp=0.1), or "lenient" (temp=0.3)
        
        Returns:
            Dict with 'verdict' and 'reason'
        """
        try:
            evidence_text = self.format_evidence(evidence_chunks)
            
            # Select prompt and temperature based on perspective
            if perspective == "strict":
                prompt_template = self.strict_prompt
                temperature = 0.0
            elif perspective == "lenient":
                prompt_template = self.lenient_prompt
                temperature = 0.0
            else:  # moderate
                prompt_template = self.moderate_prompt
                temperature = 0.0
            
            prompt_text = prompt_template.format(
                claim_text=claim.get('claim_text', ''),
                evidence_chunks=evidence_text
            )
            
            response = self.client.chat.complete(
                model="mistral-small-2503",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                timeout_ms=30000  # 30 second timeout in milliseconds
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
                
        except Exception:
            return {"verdict": "UNSUPPORTED", "reason": "API error"}


def grounded_constraint_inference(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]], perspective: str = "moderate") -> Dict[str, str]:
    """
    Standalone function for grounded constraint inference.
    
    Args:
        perspective: "strict" (temp=0.0), "moderate" (temp=0.1), or "lenient" (temp=0.3)
    
    Returns:
        Dict with 'verdict' and 'reason'
    """
    if not evidence_chunks:
        return {"verdict": "UNSUPPORTED", "reason": "No evidence chunks provided"}
    
    inference = GroundedInference()
    return inference.infer_grounded_constraint(claim, evidence_chunks, perspective)
