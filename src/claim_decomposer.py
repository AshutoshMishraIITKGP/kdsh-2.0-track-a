from typing import List, Dict
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class ClaimDecomposer:
    """Decomposes complex claims into atomic constraints."""
    
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable is required")
            
        self.client = Mistral(api_key=api_key)
        
        self.prompt_template = """Given a backstory claim, extract the minimal atomic facts it assumes.
Do NOT paraphrase. Output a bullet list.

CLAIM: {claim_text}

Extract atomic facts (one per line, starting with •):
• [atomic fact 1]
• [atomic fact 2]
• [etc.]

Keep facts simple and testable against text."""
    
    def decompose_claim(self, claim_text: str) -> List[str]:
        """
        Decompose a complex claim into atomic facts.
        
        Returns:
            List of atomic fact strings
        """
        try:
            prompt_text = self.prompt_template.format(claim_text=claim_text)
            
            response = self.client.chat.complete(
                model="mistral-small-2503",
                messages=[{"role": "user", "content": prompt_text}]
            )
            
            response_text = response.choices[0].message.content
            
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