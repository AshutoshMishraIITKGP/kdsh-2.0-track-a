from typing import List, Dict, Tuple, Set
from index_inmemory import InMemoryIndex
from semantic_index import SemanticIndex
from config import SEMANTIC_TOP_K, KEYWORD_TOP_K


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self):
        self.keyword_index = InMemoryIndex()
        self.semantic_index = SemanticIndex()
    
    def add_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """Add chunks to both indexes."""
        self.keyword_index.add_chunks(chunks)
        self.semantic_index.add_chunks(chunks)
    
    def extract_keywords(self, claim_text: str, character: str) -> List[str]:
        """Extract keywords for search (from original evidence_retrieval.py)."""
        words = claim_text.lower().replace(',', ' ').replace('.', ' ').split()
        
        keywords = [character.lower()]
        if '/' in character:
            keywords.extend([name.strip().lower() for name in character.split('/')])
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'he', 'she', 'it', 'his', 'her', 'him', 'them', 'they', 'was', 'were', 'is', 'are', 'been', 'have', 'has', 'had'}
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                keywords.append(word)
        
        return list(set(keywords))
    
    def hybrid_retrieve(self, claim: Dict[str, str], max_chunks: int = 10) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Perform hybrid retrieval combining semantic and keyword search.
        
        Returns:
            Tuple of (evidence_chunks, matching_keywords)
        """
        book_name = claim.get('book_name', '')
        character = claim.get('character', '')
        claim_text = claim.get('claim_text', '')
        
        # Semantic search
        semantic_results = self.semantic_index.semantic_search(
            claim_text, book_id=book_name, top_k=SEMANTIC_TOP_K
        )
        
        # Keyword search
        keywords = self.extract_keywords(claim_text, character)
        keyword_results = []
        matching_keywords = []
        
        for keyword in keywords[:5]:
            matches = self.keyword_index.simple_text_search(
                keyword, book_id=book_name, top_k=KEYWORD_TOP_K
            )
            if matches:
                matching_keywords.append(keyword)
                keyword_results.extend(matches)
        
        # Union and deduplicate
        seen_chunks: Set[str] = set()
        combined_results = []
        
        # Add semantic results first
        for chunk in semantic_results:
            chunk_id = chunk['chunk_id']
            if chunk_id not in seen_chunks:
                combined_results.append(chunk)
                seen_chunks.add(chunk_id)
        
        # Add keyword results
        for chunk in keyword_results:
            chunk_id = chunk['chunk_id']
            if chunk_id not in seen_chunks:
                combined_results.append(chunk)
                seen_chunks.add(chunk_id)
        
        # Sort by relative position (timeline order)
        combined_results.sort(key=lambda x: x['relative_position'])
        
        # Limit results
        if len(combined_results) > max_chunks:
            combined_results = combined_results[:max_chunks]
        
        return combined_results, matching_keywords