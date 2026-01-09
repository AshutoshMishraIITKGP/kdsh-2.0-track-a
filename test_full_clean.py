#!/usr/bin/env python3
"""
Full Test - Grounded verification + Semantic presence
"""

import sys
import csv
import json
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from semantic_index import SemanticIndex
from final_decision import aggregate_final_decision


def load_train_data():
    """Load training data from CSV."""
    train_path = Path("data/train.csv")
    claims = []
    
    with open(train_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append({
                'claim_id': row['id'],
                'book_name': row['book_name'].lower().replace(" ", "_"),
                'character': row['char'],
                'claim_text': row['content'],
                'true_label': row['label']
            })
    
    return claims


def load_cached_chunks(book_name):
    """Load cached chunks for a book."""
    cache_path = Path(f"cache/chunks/{book_name}.jsonl")
    chunks = []
    
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
    
    return chunks


def test_full_flow():
    """Test with grounded verification + semantic presence."""
    
    print("=== Full Flow: Grounded + Semantic (30 Random Claims) ===\n")
    
    # Load training data
    claims = load_train_data()
    print(f"Loaded {len(claims)} total claims")
    
    # Initialize semantic index with cached embeddings
    print("Loading semantic index...")
    semantic_index = SemanticIndex()
    
    # Load cached chunks for both books
    books = ['in_search_of_the_castaways', 'the_count_of_monte_cristo']
    
    for book_name in books:
        chunks = load_cached_chunks(book_name)
        print(f"Loaded {len(chunks)} chunks for {book_name}")
        semantic_index.add_chunks(chunks)
    
    # Test 30 random claims
    test_claims = random.sample(claims, min(30, len(claims)))
    print(f"Testing {len(test_claims)} random claims...\n")
    
    correct = 0
    total = 0
    grounded_count = 0
    semantic_only_count = 0
    
    # For precision/recall calculation
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    for i, claim in enumerate(test_claims):
        print(f"--- Claim {i+1}: ID {claim['claim_id']} ---")
        print(f"Character: {claim['character']}")
        # Handle Unicode
        claim_text = claim['claim_text'].encode('ascii', 'ignore').decode('ascii')
        print(f"Claim: {claim_text[:60]}...")
        print(f"True label: {claim['true_label']}")
        
        # Step 1: Grounded verification (retrieval)
        evidence_chunks = semantic_index.semantic_retrieve(claim, max_chunks=5)
        print(f"Retrieved: {len(evidence_chunks)} chunks")
        
        # Step 2: Final decision (grounded → semantic if needed)
        try:
            result = aggregate_final_decision(claim, evidence_chunks, semantic_index)
            predicted = result['final_decision']
            explanation = result['explanation']
            
            print(f"Predicted: {predicted}")
            print(f"Method: {result.get('method', 'UNKNOWN')}")
            print(f"Grounded: {result.get('grounded_verdict', 'N/A')}")
            print(f"Semantic: {result.get('semantic_verdict', 'N/A')}")
            print(f"Atoms: {result.get('atoms_evaluated', 0)}, Violations: {result.get('violations', 0)}")
            
            # Track method used
            method = result.get('method', 'UNKNOWN')
            if 'SEMANTIC' in method:
                semantic_only_count += 1
            else:
                grounded_count += 1
            
            if predicted != 'not_evaluable':
                is_correct = predicted == claim['true_label']
                if is_correct:
                    correct += 1
                total += 1
                
                # Track confusion matrix
                true_label = claim['true_label']
                if predicted == 'CONTRADICT' and true_label == 'CONTRADICT':
                    true_positives += 1
                elif predicted == 'CONTRADICT' and true_label == 'CONSISTENT':
                    false_positives += 1
                elif predicted == 'CONSISTENT' and true_label == 'CONTRADICT':
                    false_negatives += 1
                elif predicted == 'CONSISTENT' and true_label == 'CONSISTENT':
                    true_negatives += 1
                
                print(f"Correct: {is_correct}")
            else:
                print("Not evaluable")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    # Results
    if total > 0:
        accuracy = correct / total
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"=== RESULTS ===")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision (CONTRADICT): {precision:.2%}")
        print(f"Recall (CONTRADICT): {recall:.2%}")
        print(f"F1-Score: {f1_score:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 CONTRADICT  CONSISTENT")
        print(f"Actual CONTRADICT    {true_positives:2d}        {false_negatives:2d}")
        print(f"       CONSISTENT    {false_positives:2d}        {true_negatives:2d}")
        print(f"\nMethod Distribution:")
        print(f"Grounded decisions: {grounded_count}")
        print(f"Semantic-only decisions: {semantic_only_count}")
    else:
        print("No evaluable claims")


if __name__ == "__main__":
    test_full_flow()