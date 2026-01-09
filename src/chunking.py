from typing import List, Dict, Set
import re
import unicodedata
import csv
from pathlib import Path

from config import TARGET_TOKENS_PER_CHUNK, DATA_DIR


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token."""
    return len(text) // 4


def detect_section_boundary(paragraph: str) -> bool:
    """Detect chapter/section boundaries."""
    p = paragraph.strip().upper()
    return bool(re.match(r'^(CHAPTER|PART|BOOK|SECTION)\s+[IVXLCDM0-9]+', p) or 
                re.match(r'^[IVXLCDM]+\.|^\d+\.', p))


def normalize_text(text: str) -> str:
    """Normalize text for character matching."""
    # Remove accents/diacritics
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Lowercase, remove punctuation, collapse whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return ' '.join(text.split())


def load_character_lexicon() -> Dict[str, Set[str]]:
    """Load character names from backstory CSVs."""
    lexicon = {}
    
    for csv_file in ['train.csv', 'test.csv']:
        csv_path = DATA_DIR / csv_file
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    book_name = row['book_name']
                    char_name = row['char']
                    
                    if book_name and char_name:
                        if book_name not in lexicon:
                            lexicon[book_name] = set()
                        
                        # Normalize and add character name
                        normalized = normalize_text(char_name)
                        if normalized and len(normalized.split()) >= 1:
                            lexicon[book_name].add(normalized)
    
    return lexicon


def extract_characters(text: str, book_id: str, character_lexicon: Dict[str, Set[str]]) -> List[str]:
    """Extract character names using lexicon-based matching."""
    # Find book name from book_id (remove file extension and normalize)
    book_name = book_id.replace('.txt', '')
    
    # Get character list for this book
    if book_name not in character_lexicon:
        return []
    
    characters = character_lexicon[book_name]
    normalized_text = normalize_text(text)
    
    found_characters = []
    for char_name in characters:
        # Check if character name appears in text
        if char_name in normalized_text:
            # Verify it's not a partial match by checking word boundaries
            pattern = r'\b' + re.escape(char_name) + r'\b'
            if re.search(pattern, normalized_text):
                found_characters.append(char_name)
    
    return sorted(list(set(found_characters)))


def get_phase(relative_position: float) -> str:
    """Determine narrative phase based on relative position."""
    if relative_position < 0.3:
        return "early"
    elif relative_position < 0.7:
        return "middle"
    else:
        return "late"


def chunk_book(book: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Split a book using C-D-F-G chunking strategy.
    
    Args:
        book: {"book_id": str, "text": str}
    
    Returns:
        List of chunks with required metadata
    """
    book_id = book["book_id"]
    text = book["text"]
    
    # Load character lexicon once per book
    character_lexicon = load_character_lexicon()
    
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    overlap_size = 175  # ~150-200 tokens overlap
    
    i = 0
    while i < len(paragraphs):
        paragraph = paragraphs[i]
        para_tokens = estimate_tokens(paragraph)
        
        # Section boundary resets accumulation
        if detect_section_boundary(paragraph) and current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = [paragraph]
            current_tokens = para_tokens
        # Normal chunk size limit
        elif current_chunk and current_tokens + para_tokens > TARGET_TOKENS_PER_CHUNK:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
            
            # Create overlap for next chunk
            overlap_paras = []
            overlap_tokens = 0
            j = len(current_chunk) - 1
            
            while j >= 0 and overlap_tokens < overlap_size:
                para = current_chunk[j]
                para_tok = estimate_tokens(para)
                if overlap_tokens + para_tok <= overlap_size * 1.5:  # Allow some flexibility
                    overlap_paras.insert(0, para)
                    overlap_tokens += para_tok
                j -= 1
            
            current_chunk = overlap_paras + [paragraph]
            current_tokens = overlap_tokens + para_tokens
        else:
            current_chunk.append(paragraph)
            current_tokens += para_tokens
        
        i += 1
    
    # Add final chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunks.append(chunk_text)
    
    # Create result with all required metadata
    result = []
    total_chunks = len(chunks)
    
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{book_id}_chunk_{i:04d}"
        relative_position = i / (total_chunks - 1) if total_chunks > 1 else 0.0
        
        result.append({
            "chunk_id": chunk_id,
            "book_id": book_id,
            "text": chunk_text,
            "relative_position": relative_position,
            "phase": get_phase(relative_position),
            "mentioned_characters": extract_characters(chunk_text, book_id, character_lexicon)
        })
    
    return result