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
        
        self.prompt_template = """Given a backstory claim, extract 8-10 PLOT-CRITICAL atomic facts.
Focus on facts that can be CONTRADICTED by canon (ignore obvious/trivial facts).
Prioritize: Names, Locations, Dates, Events, Relationships, Causes.

CLAIM: {claim_text}

Extract 8-10 plot-critical atomic facts (one per line, starting with •):
• [atomic fact 1]
• [atomic fact 2]
• [etc.]

PRIORITIZE:
- Named entities (people, places, organizations)
- Specific dates/years
- Major events (arrest, death, meeting, betrayal)
- Relationships (father, brother, met X)
- Causes/reasons (because of X, triggered by Y)
- Locations (Madrid, Paris, Tasmania)

IGNORE:
- Generic facts ("He was a person", "The event happened")
- Obvious implications
- Pure adjectives without plot impact

Focus on VERIFIABLE, PLOT-CRITICAL facts that could contradict canon."""
    
    def decompose_claim(self, claim_text: str) -> List[str]:
        """
        Decompose a complex claim into 8-10 plot-critical atomic facts.
        
        Returns:
            List of plot-critical atomic fact strings (target: 8-10 facts)
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