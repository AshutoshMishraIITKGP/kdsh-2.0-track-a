from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


class GroundedInference:
    """LLM-based grounded constraint inference using bounded evidence."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required")
            
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            top_p=0.9,
            max_tokens=100,
            groq_api_key=api_key
        )
        
        self.prompt = PromptTemplate(
            template="""You are checking whether the following passages impose constraints
on the claim atom.

CLAIM ATOM: {claim_text}
PASSAGES: {evidence_chunks}

Classify as:
- HARD_VIOLATION: Atom contradicts explicit text in the passages
- UNSUPPORTED: Atom introduces detailed new facts not in passages
- NO_CONSTRAINT: Passages neither forbid nor require the atom

Output ONLY one label:
HARD_VIOLATION, UNSUPPORTED, or NO_CONSTRAINT""",
            input_variables=["claim_text", "evidence_chunks"]
        )
    
    def format_evidence(self, evidence_chunks: List[Dict[str, str]]) -> str:
        """Format evidence chunks for the prompt."""
        if not evidence_chunks:
            return "No evidence chunks found."
        
        formatted = []
        for i, chunk in enumerate(evidence_chunks, 1):
            formatted.append(f"Chunk {i}:\n{chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}")
        
        return "\n\n".join(formatted)
    
    def infer_grounded_constraint(self, claim: Dict[str, str], evidence_chunks: List[Dict[str, str]]) -> str:
        """
        Perform grounded constraint inference using LLM.
        
        Returns:
            "SUPPORTED", "INCOMPATIBLE", or "ABSENT"
        """
        try:
            evidence_text = self.format_evidence(evidence_chunks)
            
            prompt_text = self.prompt.format(
                claim_text=claim.get('claim_text', ''),
                evidence_chunks=evidence_text
            )
            
            response = self.llm.invoke(prompt_text)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract label
            response_text = response_text.strip().upper()
            if "HARD_VIOLATION" in response_text:
                return "HARD_VIOLATION"
            elif "UNSUPPORTED" in response_text:
                return "UNSUPPORTED"
            else:
                return "NO_CONSTRAINT"
                
        except Exception as e:
            print(f"Grounded inference error: {e}")
            # Default to NO_CONSTRAINT on error
            return "NO_CONSTRAINT"


def grounded_constraint_inference(claim: Dict[str, str], evidence_chunks: List[Dict[str, str]]) -> str:
    """
    Standalone function for grounded constraint inference.
    
    Returns:
        "SUPPORTED", "INCOMPATIBLE", or "ABSENT"
    """
    if not evidence_chunks:
        return "ABSENT"
    
    inference = GroundedInference()
    return inference.infer_grounded_constraint(claim, evidence_chunks)