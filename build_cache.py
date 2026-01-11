#!/usr/bin/env python3
"""
Cache Builder - Pre-compute chunks and embeddings
"""

import sys
import json
from pathlib import Path
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_books import load_all_books
from chunking import chunk_book


def create_cache_structure():
    """Create cache directory structure."""
    cache_dirs = [
        "cache/chunks",
        "cache/embeddings/e5-large-v2"
    ]
    
    for dir_path in cache_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("OK Cache structure created")


def cache_chunks():
    """Cache book chunks as JSONL files."""
    print("\n1. Caching chunks...")
    
    books = load_all_books()
    cached_books = {}
    
    for book in books:
        book_id = book['book_id']
        print(f"   Processing {book_id}...")
        
        # Generate chunks
        chunks = chunk_book(book)
        
        # Save as JSONL
        cache_path = Path(f"cache/chunks/{book_id}.jsonl")
        with open(cache_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
        
        cached_books[book_id] = {
            'chunk_count': len(chunks),
            'cache_path': str(cache_path)
        }
        
        print(f"     OK Cached {len(chunks)} chunks")
    
    return cached_books


def cache_embeddings(cached_books):
    """Cache embeddings using E5 model."""
    print("\n2. Caching embeddings...")
    
    # Initialize E5 model with GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer("intfloat/e5-large-v2", device=device)
    print(f"   Using device: {device}")
    
    for book_id, book_info in cached_books.items():
        print(f"   Processing embeddings for {book_id}...")
        
        # Load chunks
        chunks = []
        with open(book_info['cache_path'], 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
        
        # Generate embeddings with E5 prefixes
        texts = [f"passage: {chunk['text']}" for chunk in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save FAISS index
        faiss_path = f"cache/embeddings/e5-large-v2/{book_id}.faiss"
        faiss.write_index(index, faiss_path)
        
        # Save chunk mapping
        mapping_path = f"cache/embeddings/e5-large-v2/{book_id}_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump([chunk['chunk_id'] for chunk in chunks], f)
        
        print(f"     OK Cached {len(embeddings)} embeddings")


def build_cache():
    """Build complete cache system."""
    print("=== Building Cache System ===")
    
    # Create structure
    create_cache_structure()
    
    # Cache components
    cached_books = cache_chunks()
    cache_embeddings(cached_books)
    
    print("\n=== Cache Build Complete ===")
    print("OK Chunks cached")
    print("OK E5 embeddings cached")
    print("\nCache ready for fast testing!")


if __name__ == "__main__":
    build_cache()
