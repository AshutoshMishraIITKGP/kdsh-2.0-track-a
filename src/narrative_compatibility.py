from typing import Dict, List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


class NarrativeCompatibilityInference:
    """Evaluates narrative compatibility for claims marked as ABSENT."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required")
            
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            top_p=0.9,
            max_tokens=200,
            groq_api_key=api_key
        )
        
        self.prompt = PromptTemplate(
            template="""You are evaluating whether a backstory claim is NARRATIVELY COMPATIBLE
with a novel, even if it is not explicitly stated.

Rules:
- Do NOT require explicit textual evidence.
- The claim must not contradict known facts, character traits, or events.
- Plausible extrapolation from the character's role, background, and setting is allowed.
- If the claim introduces facts that would likely conflict with the story,
  mark it INCOMPATIBLE.
- If the claim fits naturally into the narrative without contradiction,
  mark it COMPATIBLE.

IMPORTANT — IMPLICIT NARRATIVE CONTRADICTIONS

A claim should be marked INCOMPATIBLE if it:

• Assigns motivations, intentions, or moral alignment that conflict with the
  character's established role or arc, even if not explicitly negated.

• Reframes a character's actions in a way that undermines known themes,
  narrative stakes, or power dynamics of the story.

• Attributes relationships, loyalties, or betrayals that would change the
  meaning of key events in the novel.

• Makes the character act "out of character" relative to how they are portrayed,
  even if the text does not explicitly forbid the claim.

Do NOT treat all plausible extrapolations as compatible.
Plausibility alone is insufficient — narrative fidelity matters.

You are judging narrative compatibility, not factual truth.

CLAIM: {claim_text}
CHARACTER: {character}
BOOK CONTEXT: {book_context}

Output ONLY one label:
- COMPATIBLE
- INCOMPATIBLE""",
            input_variables=["claim_text", "character", "book_context"]
        )
    
    def evaluate_compatibility(self, claim: Dict[str, str], book_context: str) -> str:
        """
        Evaluate narrative compatibility for an ABSENT claim.
        
        Returns:
            "COMPATIBLE" or "INCOMPATIBLE"
        """
        try:
            prompt_text = self.prompt.format(
                claim_text=claim.get('claim_text', ''),
                character=claim.get('character', ''),
                book_context=book_context
            )
            
            response = self.llm.invoke(prompt_text)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract label
            response_text = response_text.strip().upper()
            if "COMPATIBLE" in response_text and "INCOMPATIBLE" not in response_text:
                return "COMPATIBLE"
            elif "INCOMPATIBLE" in response_text:
                return "INCOMPATIBLE"
            else:
                # Default to INCOMPATIBLE if unclear
                return "INCOMPATIBLE"
                
        except Exception:
            # Default to INCOMPATIBLE on error
            return "INCOMPATIBLE"


def narrative_compatibility_inference(claim: Dict[str, str], book_context: str) -> str:
    """
    Standalone function for narrative compatibility evaluation.
    
    Returns:
        "COMPATIBLE" or "INCOMPATIBLE"
    """
    evaluator = NarrativeCompatibilityInference()
    return evaluator.evaluate_compatibility(claim, book_context)