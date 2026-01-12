#!/usr/bin/env python3
"""
Test Set Evaluation - Final Model (15 chunks with character/temporal boosting)
Format: story_id, prediction (1=consistent, 0=inconsistent), rationale
"""

import sys
import csv
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from semantic_index import SemanticIndex
from final_decision_ensemble import aggregate_final_decision
import json


def load_test_data():
    """Load test data from CSV."""
    test_path = Path("data/test.csv")
    claims = []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append({
                'claim_id': row['id'],
                'book_name': row['book_name'].lower().replace(" ", "_"),
                'character': row['char'],
                'claim_text': row['content']
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


def generate_rationale(result):
    """Generate rationale from dual agent evaluation."""
    method = result.get('method', 'UNKNOWN')
    explanation = result.get('explanation', 'No explanation')
    atom_details = result.get('atom_details', [])
    
    if not atom_details:
        return explanation
    
    # Count verdicts
    supported = sum(1 for d in atom_details if d.get('verdict') == 'SUPPORTED')
    violations = sum(1 for d in atom_details if d.get('verdict') == 'HARD_VIOLATION')
    unsupported = sum(1 for d in atom_details if d.get('verdict') == 'UNSUPPORTED')
    
    return f"{explanation}; Atoms: {len(atom_details)} (supported={supported}, violations={violations}, unsupported={unsupported})"


def run_test():
    """Process test set with dual agent system."""
    
    print("=== Final Model Test Evaluation (15 chunks, 1-3-6 batches) ===")
    print("Output: story_id, prediction (1=consistent, 0=inconsistent), rationale\n")
    
    # Load test data
    claims = load_test_data()
    print(f"Loaded {len(claims)} test claims")
    
    # Initialize semantic index
    print("Loading semantic index...")
    semantic_index = SemanticIndex()
    
    # Load cached chunks
    books = ['in_search_of_the_castaways', 'the_count_of_monte_cristo']
    
    for book_name in books:
        chunks = load_cached_chunks(book_name)
        print(f"Loaded {len(chunks)} chunks for {book_name}")
        semantic_index.add_chunks(chunks)
    
    print(f"\nProcessing {len(claims)} test claims...\n")
    
    results = []
    correct = 0
    total = 0
    
    for i, claim in enumerate(claims, 1):
        print(f"--- Claim {i}: ID {claim['claim_id']} ---")
        print(f"Character: {claim['character']}")
        claim_text = claim['claim_text'].encode('ascii', 'ignore').decode('ascii')
        print(f"Claim: {claim_text[:60]}...")
        
        try:
            # Retrieve 15 chunks with character/temporal boosting
            evidence_chunks = semantic_index.semantic_retrieve(claim, max_chunks=15)
            print(f"Retrieved: {len(evidence_chunks)} chunks")
            
            # Final model decision (3-perspective ensemble)
            result = aggregate_final_decision(claim, evidence_chunks, semantic_index)
            predicted = result['final_decision'].upper()
            
            # Convert to binary (1=consistent, 0=inconsistent)
            prediction = 1 if predicted == 'CONSISTENT' else 0
            
            # Generate rationale
            rationale = generate_rationale(result)
            
            print(f"Predicted: {predicted}")
            print(f"Method: {result.get('method', 'UNKNOWN')}")
            print(f"Explanation: {result.get('explanation', 'No explanation')}")
            
            results.append({
                'story_id': claim['claim_id'],
                'prediction': prediction,
                'rationale': rationale
            })
            
            total += 1
            print()
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'story_id': claim['claim_id'],
                'prediction': 1,
                'rationale': f'Error: {str(e)}'
            })
            print()
    
    # Write results
    output_path = Path("results.csv")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['story_id', 'prediction', 'rationale'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n=== RESULTS ===")
    print(f"Total: {len(results)} predictions")
    consistent = sum(1 for r in results if r['prediction'] == 1)
    inconsistent = len(results) - consistent
    print(f"Distribution: {consistent} consistent ({consistent/len(results)*100:.1f}%), {inconsistent} inconsistent ({inconsistent/len(results)*100:.1f}%)")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_test()