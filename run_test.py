#!/usr/bin/env python3
"""
Test Set Evaluation - Generates results.csv for submission
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


def generate_rationale(result, claim_text):
    """Generate detailed rationale from decision result."""
    decision = result['final_decision'].upper()
    violation_atoms = result.get('violation_atoms', [])
    atoms_evaluated = result.get('atoms_evaluated', 0)
    
    if decision == 'CONTRADICT':
        if violation_atoms and len(violation_atoms) > 0:
            # Get first violation with reason
            first_violation = violation_atoms[0]
            if isinstance(first_violation, dict):
                atom = first_violation.get('atom', 'Unknown')
                reason = first_violation.get('reason', 'No reason provided')
                return f"{atom} - {reason}"
            else:
                return f"Violation: {first_violation}"
        else:
            return "Narrative inconsistency detected"
    else:
        return f"All {atoms_evaluated} atomic claims supported by narrative evidence"


def run_test():
    """Process test set and generate results.csv."""
    
    print("=== Test Set Evaluation ===")
    print("Output format: story_id, prediction (1=consistent, 0=inconsistent), rationale\n")
    
    # Load test data
    claims = load_test_data()
    print(f"Loaded {len(claims)} test claims")
    
    # Initialize semantic index with cached embeddings
    print("Loading semantic index...")
    semantic_index = SemanticIndex()
    
    # Load cached chunks for both books
    books = ['in_search_of_the_castaways', 'the_count_of_monte_cristo']
    
    for book_name in books:
        chunks = load_cached_chunks(book_name)
        print(f"Loaded {len(chunks)} chunks for {book_name}")
        semantic_index.add_chunks(chunks)
    
    print(f"\nProcessing {len(claims)} test claims...\n")
    
    # Process each claim
    results = []
    
    for i, claim in enumerate(claims, 1):
        print(f"Processing claim {i}/{len(claims)}: {claim['claim_id']}")
        
        try:
            # Retrieve evidence
            evidence_chunks = semantic_index.semantic_retrieve(claim, max_chunks=10)
            
            # Get prediction
            result = aggregate_final_decision(claim, evidence_chunks, semantic_index)
            predicted = result['final_decision'].upper()
            
            # Convert to binary format (1=consistent, 0=inconsistent)
            prediction = 1 if predicted == 'CONSISTENT' else 0
            
            # Generate rationale
            rationale = generate_rationale(result, claim['claim_text'])
            
            # Store result
            results.append({
                'story_id': claim['claim_id'],
                'prediction': prediction,
                'rationale': rationale
            })
            
            print(f"  Predicted: {prediction} ({'CONSISTENT' if prediction == 1 else 'INCONSISTENT'})")
            print(f"  Rationale: {rationale}\n")
            
        except Exception as e:
            print(f"  Error: {e}")
            # Default to consistent on error
            results.append({
                'story_id': claim['claim_id'],
                'prediction': 1,
                'rationale': 'Error during evaluation - defaulted to consistent'
            })
    
    # Write results to CSV
    output_path = Path("results.csv")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['story_id', 'prediction', 'rationale'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n=== Results saved to {output_path} ===")
    print(f"Total predictions: {len(results)}")
    print(f"Format: story_id, prediction (1=consistent, 0=inconsistent), rationale")


if __name__ == "__main__":
    run_test()
