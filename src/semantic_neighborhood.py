from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


class SemanticNeighborhoodEvaluator:
    """Evaluates semantic compatibility using narrative neighborhoods."""
    
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
            template="""Analyze if these passages contradict the claim about the character.

CLAIM: {claim_text}

PASSAGES: {semantic_passages}

CRITICAL RULE: Do NOT treat lack of mention as contradiction.
Only mark INCOMPATIBLE if the text actively forbids the claim.

Look for:
- Direct contradictions of the claim
- Character doing the opposite of what's claimed
- Events that make the claim impossible
- Clear inconsistencies with the claim

Only answer INCOMPATIBLE if you find clear contradictory evidence.
If passages are neutral, unrelated, or don't address the claim, answer COMPATIBLE.

Answer: COMPATIBLE or INCOMPATIBLE""",
            input_variables=["claim_text", "semantic_passages"]
        )
    
    def format_passages(self, semantic_chunks: List[Dict[str, str]]) -> str:
        """Format semantic neighborhood passages for the prompt."""
        if not semantic_chunks:
            return "No semantically related passages found."
        
        formatted = []
        for i, chunk in enumerate(semantic_chunks, 1):
            formatted.append(f"Passage {i}:\n{chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}")
        
        return "\n\n".join(formatted)
    
    def evaluate_semantic_compatibility(self, claim: Dict[str, str], semantic_chunks: List[Dict[str, str]]) -> str:
        """
        Evaluate semantic compatibility using narrative neighborhoods.
        
        Returns:
            "COMPATIBLE" or "INCOMPATIBLE"
        """
        try:
            passages_text = self.format_passages(semantic_chunks)
            
            prompt_text = self.prompt.format(
                claim_text=claim.get('claim_text', ''),
                semantic_passages=passages_text
            )
            
            response = self.llm.invoke(prompt_text)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Be more strict about parsing - look for explicit contradiction signals
            response_text = response_text.strip().upper()
            
            # Look for strong incompatibility signals
            incompatible_signals = ["INCOMPATIBLE", "CONTRADICT", "OPPOSITE", "IMPOSSIBLE"]
            compatible_signals = ["COMPATIBLE", "CONSISTENT", "NEUTRAL"]
            
            if any(signal in response_text for signal in incompatible_signals):
                # Double-check it's not saying "not incompatible"
                if "NOT INCOMPATIBLE" not in response_text and "NOT CONTRADICT" not in response_text:
                    return "INCOMPATIBLE"
            
            if any(signal in response_text for signal in compatible_signals):
                return "COMPATIBLE"
            
            # Default to compatible if unclear
            return "COMPATIBLE"
                
        except Exception:
            # Default to COMPATIBLE on error (conservative)
            return "COMPATIBLE"


def semantic_neighborhood_evaluation(claim: Dict[str, str], semantic_chunks: List[Dict[str, str]]) -> str:
    """
    Standalone function for semantic neighborhood evaluation.
    
    Returns:
        "COMPATIBLE" or "INCOMPATIBLE"
    """
    evaluator = SemanticNeighborhoodEvaluator()
    return evaluator.evaluate_semantic_compatibility(claim, semantic_chunks)