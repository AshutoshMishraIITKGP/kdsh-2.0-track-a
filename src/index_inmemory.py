from typing import List, Dict, Optional
from config import DEFAULT_TOP_K


class InMemoryIndex:
    """Timeline-aware narrative memory for chunk retrieval."""
    
    def __init__(self):
        self.chunks: List[Dict[str, str]] = []
    
    def add_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """Add chunks to the index."""
        self.chunks.extend(chunks)
    
    def get_chunks_by_book(self, book_id: str) -> List[Dict[str, str]]:
        """Retrieve all chunks for a specific book in chronological order."""
        book_chunks = [chunk for chunk in self.chunks if chunk["book_id"] == book_id]
        return sorted(book_chunks, key=lambda x: x["relative_position"])
    
    def get_chunks_by_position_range(self, book_id: str, min_pos: float = 0.0, max_pos: float = 1.0) -> List[Dict[str, str]]:
        """Retrieve chunks within a relative position range for timeline-aware access."""
        book_chunks = self.get_chunks_by_book(book_id)
        return [chunk for chunk in book_chunks 
                if min_pos <= chunk["relative_position"] <= max_pos]
    
    def simple_text_search(self, query: str, book_id: Optional[str] = None, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, str]]:
        """Simple text-based retrieval (case-insensitive substring matching)."""
        query_lower = query.lower()
        
        # Filter by book if specified
        search_chunks = self.get_chunks_by_book(book_id) if book_id else self.chunks
        
        # Find chunks containing query text
        matches = []
        for chunk in search_chunks:
            if query_lower in chunk["text"].lower():
                matches.append(chunk)
        
        # Sort by relative position to maintain chronological order
        matches.sort(key=lambda x: (x["book_id"], x["relative_position"]))
        
        return matches[:top_k]
    
    def get_all_books(self) -> List[str]:
        """Get list of all book IDs in the index."""
        return list(set(chunk["book_id"] for chunk in self.chunks))
    
    def get_chunk_count(self, book_id: Optional[str] = None) -> int:
        """Get total chunk count, optionally filtered by book."""
        if book_id:
            return len(self.get_chunks_by_book(book_id))
        return len(self.chunks)