from typing import List, Dict
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


def collect_character_chunks(chunks: List[Dict[str, str]], character_name: str) -> List[Dict[str, str]]:
    """
    Return all chunks where the character name appears literally.
    Case-insensitive. No embeddings. No inference.
    """
    character_lower = character_name.lower()
    character_chunks = []
    
    for chunk in chunks:
        if character_lower in chunk['text'].lower():
            character_chunks.append(chunk)
    
    return character_chunks


class CharacterProfileGenerator:
    """Generates cached character narrative profiles."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required")
            
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            top_p=0.9,
            max_tokens=400,
            groq_api_key=api_key
        )
        
        self.prompt = PromptTemplate(
            template="""You are creating a CHARACTER NARRATIVE PROFILE from a novel.

Your task is to summarize ONLY what is supported by the provided text.
Do NOT invent facts.
Do NOT speculate beyond the text.
Do NOT add backstory details that are not present.

Focus on:
• the character's role in the story
• personality traits explicitly or repeatedly implied
• narrative function (hero, guide, antagonist, comic relief, etc.)
• typical actions and behavior patterns
• known limitations or things the character does NOT do
• relationships as described (do not invent new ones)
• thematic alignment

Write in neutral, analytical language.
This profile will be used to judge narrative compatibility of future claims.

TEXT:
{character_chunks}

OUTPUT FORMAT:

CHARACTER PROFILE — {character_name}

Role:
Traits:
Narrative function:
Typical actions:
Known limitations:
Key relationships:
Thematic alignment:""",
            input_variables=["character_chunks", "character_name"]
        )
    
    def build_character_profile(self, character_name: str, character_chunks: List[Dict[str, str]]) -> str:
        """
        Generate a narrative profile strictly from the provided chunks.
        """
        if not character_chunks:
            return f"CHARACTER PROFILE — {character_name}\n\nNo textual evidence found for this character."
        
        # Combine chunk texts
        chunks_text = "\n\n".join([f"Chunk {i+1}: {chunk['text']}" for i, chunk in enumerate(character_chunks)])
        
        try:
            prompt_text = self.prompt.format(
                character_chunks=chunks_text,
                character_name=character_name
            )
            
            response = self.llm.invoke(prompt_text)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return response_text.strip()
            
        except Exception as e:
            return f"CHARACTER PROFILE — {character_name}\n\nError generating profile: {str(e)}"


class CharacterProfileCache:
    """Manages persistent caching of character profiles."""
    
    def __init__(self, cache_dir: str = "profiles"):
        self.cache_dir = Path(cache_dir)
        self.generator = CharacterProfileGenerator()
    
    def _get_profile_path(self, book_name: str, character_name: str) -> Path:
        """Get the file path for a character profile."""
        book_dir = self.cache_dir / book_name
        book_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize character name for filename
        safe_name = "".join(c for c in character_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        return book_dir / f"{safe_name}.txt"
    
    def _get_chunks_hash(self, chunks: List[Dict[str, str]]) -> str:
        """Generate hash of chunks for cache invalidation."""
        chunks_text = "".join(chunk['text'] for chunk in chunks)
        return hashlib.md5(chunks_text.encode()).hexdigest()[:8]
    
    def get_character_profile(self, book_name: str, character_name: str, all_chunks: List[Dict[str, str]]) -> str:
        """
        Get character profile, generating if not cached or if source changed.
        """
        profile_path = self._get_profile_path(book_name, character_name)
        
        # Collect character chunks
        character_chunks = collect_character_chunks(all_chunks, character_name)
        current_hash = self._get_chunks_hash(character_chunks)
        
        # Check if cached profile exists and is current
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    cached_content = f.read()
                
                # Check if hash matches (stored in first line as comment)
                if cached_content.startswith(f"# Hash: {current_hash}"):
                    return cached_content[len(f"# Hash: {current_hash}\n"):]
            except Exception:
                pass
        
        # Generate new profile
        profile = self.generator.build_character_profile(character_name, character_chunks)
        
        # Cache the profile with hash
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                f.write(f"# Hash: {current_hash}\n{profile}")
        except Exception:
            pass  # Continue even if caching fails
        
        return profile


# Global cache instance
_profile_cache = CharacterProfileCache()


def get_character_profile(book_name: str, character_name: str, all_chunks: List[Dict[str, str]]) -> str:
    """
    Get cached character profile for narrative compatibility inference.
    """
    return _profile_cache.get_character_profile(book_name, character_name, all_chunks)