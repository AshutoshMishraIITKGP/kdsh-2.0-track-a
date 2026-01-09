#!/usr/bin/env python3
"""
Cache Builder - Pre-compute chunks, embeddings, and profiles
"""

import sys
import json
import os
from pathlib import Path
import faiss
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_books import load_all_books
from chunking import chunk_book
from semantic_index import SemanticIndex
from character_profiles import get_character_profile
from sentence_transformers import SentenceTransformer


def normalize_book_name(name):
    return name.lower().replace(" ", "_").strip()


def create_cache_structure():
    """Create cache directory structure."""
    cache_dirs = [
        "cache/chunks",
        "cache/embeddings/e5-large-v2", 
        "cache/retrieval",
        "cache/profiles"
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
        book_id = normalize_book_name(book['book_id'])
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
        
        print(f"     Cached {len(chunks)} chunks")
    
    return cached_books


def cache_embeddings(cached_books):
    """Cache embeddings using E5 model."""
    print("\n2. Caching embeddings...")
    
    # Initialize E5 model with GPU if available
    import torch
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
        embeddings = model.encode(texts, convert_to_numpy=True)
        
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
        
        print(f"     Cached {len(embeddings)} embeddings")


def cache_profiles(cached_books):
    """Cache character profiles."""
    print("\n3. Caching character profiles...")
    
    # Common characters to profile
    characters_by_book = {
        'the_count_of_monte_cristo': [
            'Edmund Dantes', 'Fernand Mondego', 'Mercedes', 'Danglars', 
            'Abbe Faria', 'Noirtier', 'Villefort'
        ],
        'in_search_of_the_castaways': [
            'Jacques Paganel', 'Tom Ayrton', 'Thalcave', 'Kai-Koumou',
            'Lord Glenarvan', 'Mary Grant', 'Robert Grant'
        ]
    }
    
    for book_id, book_info in cached_books.items():
        print(f"   Processing profiles for {book_id}...")
        
        # Load chunks
        chunks = []
        with open(book_info['cache_path'], 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
        
        # Create book profile directory
        profile_dir = Path(f"cache/profiles/{book_id}")
        profile_dir.mkdir(exist_ok=True)
        
        # Generate profiles for known characters
        characters = characters_by_book.get(book_id, [])
        for character in characters:
            print(f"     Generating profile for {character}...")
            
            try:
                profile = get_character_profile(book_id, character, chunks)
                
                # Save profile
                safe_name = "".join(c for c in character if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_name = safe_name.replace(' ', '_').lower()
                
                profile_path = profile_dir / f"{safe_name}.txt"
                with open(profile_path, 'w', encoding='utf-8') as f:
                    f.write(profile)
                
                print(f"       OK Cached profile ({len(profile)} chars)")
                
            except Exception as e:
                print(f"       X Error: {e}")


def load_train_claims():
    """Load training claims for retrieval caching."""
    import csv
    
    claims = []
    with open("data/train.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append({
                'claim_id': row['id'],
                'book_name': normalize_book_name(row['book_name']),
                'character': row['char'],
                'claim_text': row['content'],
                'true_label': row['label']
            })
    
    return claims


def cache_retrieval_results():
    """Cache retrieval results for training claims."""
    print("\n4. Caching retrieval results...")
    
    claims = load_train_claims()
    
    # Initialize retrieval components
    from hybrid_retrieval import HybridRetriever
    from index_inmemory import InMemoryIndex
    
    keyword_index = InMemoryIndex()
    hybrid_retrieval = HybridRetriever()
    
    # Load all cached chunks
    all_chunks = []
    for book_file in Path("cache/chunks").glob("*.jsonl"):
        with open(book_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_chunks.append(json.loads(line))
    
    # Build indexes
    keyword_index.add_chunks(all_chunks)
    hybrid_retrieval.add_chunks(all_chunks)
    
    # Cache retrieval for each claim
    for i, claim in enumerate(claims[:20]):  # Cache first 20 for testing
        print(f"   Caching retrieval for claim {claim['claim_id']}...")
        
        try:
            evidence_chunks, matching_keywords = hybrid_retrieval.hybrid_retrieve(claim, max_chunks=5)
            
            retrieval_result = {
                'claim': claim,
                'evidence_chunks': evidence_chunks,
                'matching_keywords': matching_keywords,
                'evidence_count': len(evidence_chunks)
            }
            
            # Save retrieval result
            cache_path = f"cache/retrieval/train_claim_{claim['claim_id']}.json"
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(retrieval_result, f, indent=2)
            
            print(f"     OK Cached {len(evidence_chunks)} evidence chunks")
            
        except Exception as e:
            print(f"     X Error: {e}")


def build_cache():
    """Build complete cache system."""
    print("=== Building Cache System ===")
    
    # Create structure
    create_cache_structure()
    
    # Cache components
    cached_books = cache_chunks()
    cache_embeddings(cached_books)
    cache_profiles(cached_books)
    cache_retrieval_results()
    
    print("\n=== Cache Build Complete ===")
    print("OK Chunks cached")
    print("OK E5 embeddings cached") 
    print("OK Character profiles cached")
    print("OK Retrieval results cached")
    print("\nCache ready for fast testing!")


if __name__ == "__main__":
    build_cache()