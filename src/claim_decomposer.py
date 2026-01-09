from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


class ClaimDecomposer:
    """Decomposes complex claims into atomic constraints."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required")
            
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=200,
            groq_api_key=api_key
        )
        
        self.prompt = PromptTemplate(
            template="""Given a backstory claim, extract the minimal atomic facts it assumes.
Do NOT paraphrase. Output a bullet list.

CLAIM: {claim_text}

Extract atomic facts (one per line, starting with •):
• [atomic fact 1]
• [atomic fact 2]
• [etc.]

Keep facts simple and testable against text.""",
            input_variables=["claim_text"]
        )
    
    def decompose_claim(self, claim_text: str) -> List[str]:
        """
        Decompose a complex claim into atomic facts.
        
        Returns:
            List of atomic fact strings
        """
        try:
            prompt_text = self.prompt.format(claim_text=claim_text)
            response = self.llm.invoke(prompt_text)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse bullet points
            atoms = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    atom = line[1:].strip()
                    if atom:
                        atoms.append(atom)
            
            # Fallback: if no bullets found, return original claim
            return atoms if atoms else [claim_text]
            
        except Exception as e:
            print(f"Claim decomposition error: {e}")
            # Fallback to original claim
            return [claim_text]


def decompose_claim(claim_text: str) -> List[str]:
    """
    Standalone function for claim decomposition.
    
    Returns:
        List of atomic facts
    """
    decomposer = ClaimDecomposer()
    return decomposer.decompose_claim(claim_text)