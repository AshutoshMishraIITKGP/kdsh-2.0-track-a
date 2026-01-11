from typing import List, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path
from config import EMBEDDING_MODEL


class SemanticIndex:
    """Semantic search index for novel chunks using cached FAISS."""
    
    def __init__(self):
        # Delay model loading until actually needed
        self._model = None
        self.indices = {}  # book_name -> faiss index
        self.chunks_by_book = {}  # book_name -> chunks
        self.mappings = {}  # book_name -> chunk mapping
    
    @property
    def model(self):
        if self._model is None:
            print("Loading E5-large-v2 model...")
            self._model = SentenceTransformer("intfloat/e5-large-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
        return self._model
    
    def load_cached_index(self, book_name: str) -> bool:
        """Load cached FAISS index and mapping for a book."""
        cache_dir = Path(f"cache/embeddings/e5-large-v2")
        
        # Try both original book_name and normalized version
        possible_names = [book_name, book_name.lower().replace(" ", "_")]
        
        for name in possible_names:
            index_path = cache_dir / f"{name}.faiss"
            mapping_path = cache_dir / f"{name}_mapping.json"
            
            if index_path.exists() and mapping_path.exists():
                try:
                    # Load FAISS index
                    self.indices[book_name] = faiss.read_index(str(index_path))
                    
                    # Load chunk mapping
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        self.mappings[book_name] = json.load(f)
                    
                    return True
                except Exception:
                    continue
        
        return False
    
    def add_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """Add chunks using cached embeddings if available."""
        if not chunks:
            return
        
        # Group chunks by book
        books = {}
        for chunk in chunks:
            book_id = chunk.get('book_id', 'unknown')
            if book_id not in books:
                books[book_id] = []
            books[book_id].append(chunk)
        
        # Load cached indices for each book
        for book_name, book_chunks in books.items():
            self.chunks_by_book[book_name] = book_chunks
            print(f"Loading cached index for {book_name}...")
            
            if not self.load_cached_index(book_name):
                print(f"Cache miss for {book_name}, building from scratch...")
                # Fallback: build index from scratch
                self._build_index_from_scratch(book_name, book_chunks)
            else:
                print(f"Loaded cached index for {book_name}")
    
    def _build_index_from_scratch(self, book_name: str, chunks: List[Dict[str, str]]) -> None:
        """Fallback: build FAISS index from scratch."""
        texts = [f"passage: {chunk['text']}" for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.indices[book_name] = index
        self.mappings[book_name] = {str(i): i for i in range(len(chunks))}
    
    def semantic_search(self, query: str, book_id: str = None, top_k: int = 5) -> List[Dict[str, str]]:
        """Perform semantic similarity search."""
        if book_id and book_id not in self.indices:
            return []
        
        # Encode query
        query_embedding = self.model.encode([f"query: {query}"], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in specific book or all books
        if book_id:
            return self._search_book(query_embedding, book_id, top_k)
        else:
            # Search all books
            all_results = []
            for book in self.indices.keys():
                results = self._search_book(query_embedding, book, top_k)
                all_results.extend(results)
            
            # Sort by similarity and return top_k
            return all_results[:top_k]
    
    def _search_book(self, query_embedding: np.ndarray, book_name: str, top_k: int) -> List[Dict[str, str]]:
        """Search within a specific book."""
        if book_name not in self.indices or book_name not in self.chunks_by_book:
            return []
        
        index = self.indices[book_name]
        chunks = self.chunks_by_book[book_name]
        
        scores, indices = index.search(query_embedding, min(top_k, len(chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(chunks):
                results.append(chunks[idx])
        
        return results
    
    def semantic_retrieve(self, claim: Dict[str, str], max_chunks: int = 30) -> List[Dict[str, str]]:
        """Retrieve evidence chunks for grounded verification with character filtering."""
        claim_text = claim.get('claim_text', '')
        book_name = claim.get('book_name', '')
        character = claim.get('character', '')
        
        # Map normalized book names to actual book_ids in chunks
        book_mapping = {
            'the_count_of_monte_cristo': 'The Count of Monte Cristo',
            'in_search_of_the_castaways': 'In search of the castaways'
        }
        
        book_id = book_mapping.get(book_name)
        if not book_id or book_id not in self.chunks_by_book:
            return []
        
        # Get all chunks for the book
        all_results = self.semantic_search(claim_text, book_id=book_id, top_k=max_chunks * 3)
        
        # Filter by character presence
        character_filtered = []
        character_names = [character.lower()]
        if '/' in character:
            character_names.extend([name.strip().lower() for name in character.split('/')])
        
        for chunk in all_results:
            text_lower = chunk['text'].lower()
            if any(name in text_lower for name in character_names):
                character_filtered.append(chunk)
                if len(character_filtered) >= max_chunks:
                    break
        
        # If character filtering yields too few results, fall back to semantic only
        if len(character_filtered) < max_chunks // 2:
            return all_results[:max_chunks]
        
        return character_filtered
    
    def semantic_neighborhood_retrieve(self, claim: Dict[str, str], top_k: int = 20) -> List[Dict[str, str]]:
        """Retrieve semantic neighborhood for claim-centric evaluation (15-25 chunks)."""
        claim_text = claim.get('claim_text', '')
        book_name = claim.get('book_name', '')
        
        # Map normalized book names to actual book_ids in chunks
        book_mapping = {
            'the_count_of_monte_cristo': 'The Count of Monte Cristo',
            'in_search_of_the_castaways': 'In search of the castaways'
        }
        
        book_id = book_mapping.get(book_name)
        if not book_id or book_id not in self.chunks_by_book:
            return []
            
        return self.semantic_search(claim_text, book_id=book_id, top_k=top_k)