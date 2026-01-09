import unicodedata
import re


def normalize_text(text: str) -> str:
    """
    Normalize text to handle encoding issues and ensure consistent matching.
    
    Args:
        text: Input text that may have encoding issues
    
    Returns:
        Normalized text with consistent character encoding
    """
    if not text:
        return text
    
    # Unicode NFC normalization first
    text = unicodedata.normalize('NFC', text)
    
    # Handle common encoding corruptions and convert to ASCII-friendly
    replacements = {
        # Common French characters
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'á': 'a', 'â': 'a', 'ä': 'a',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o',
        'ç': 'c', 'ñ': 'n',
        
        # Common corrupted characters we saw
        '鑚': 'es', '穰': 'at',  # Dantès, Château
        
        # Other common issues
        ''': "'", ''': "'", '"': '"', '"': '"',
        '–': '-', '—': '-',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fallback: convert any remaining non-ASCII to closest ASCII
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if ord(c) < 128 or c.isspace())
    
    return text