from pathlib import Path
from typing import List, Dict
import re

from config import BOOKS_DIR
from text_normalization import normalize_text


def remove_gutenberg_metadata(text: str) -> str:
    """
    Remove Project Gutenberg header and footer from text.
    """
    # Remove header - everything before the actual story starts
    # Look for the title or first chapter
    
    # First, remove the initial Gutenberg header
    text = re.sub(r'\*\*\* START OF.*?\*\*\*.*?(?=\n)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove Project Gutenberg notes and metadata at the beginning
    text = re.sub(r'^.*?Note: Project Gutenberg.*?(?=\n\n[A-Z])', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining metadata lines at the start
    lines = text.split('\n')
    start_idx = 0
    
    # Find where the actual story content begins
    for i, line in enumerate(lines):
        line = line.strip()
        # Look for story title or chapter markers
        if (line and 
            (line.isupper() and len(line) > 5) or  # Title in caps
            line.startswith('CHAPTER') or 
            line.startswith('Chapter') or
            'Voyage Round the World' in line or
            'IN SEARCH OF' in line):
            start_idx = i
            break
    
    # Remove footer - everything after the story ends
    text = '\n'.join(lines[start_idx:])
    text = re.sub(r'\*\*\* END OF.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()
    
    return text


def load_all_books() -> List[Dict[str, str]]:
    """
    Load all .txt novels from the books directory.

    Returns a list of dicts:
    {
        "book_id": "<filename_without_extension>",
        "text": "<full novel text>"
    }
    """
    books = []

    if not BOOKS_DIR.exists():
        raise FileNotFoundError(f"Books directory not found: {BOOKS_DIR}")

    for book_path in sorted(BOOKS_DIR.glob("*.txt")):
        try:
            text = book_path.read_text(encoding="utf-8-sig")
            text = text.replace("\ufeff", "")  # extra BOM safety
            
            # Remove Project Gutenberg metadata
            text = remove_gutenberg_metadata(text)
            
            # Normalize text encoding
            text = normalize_text(text)

            books.append({
                "book_id": book_path.stem,
                "text": text
            })

        except Exception as e:
            print(f"[WARN] Failed to load {book_path.name}: {e}")

    if not books:
        raise RuntimeError("No books were loaded. Check data/raw/books.")

    return books