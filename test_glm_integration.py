#!/usr/bin/env python3
"""
Test GLM-4.7 integration
"""

import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

def test_glm_connection():
    """Test basic GLM-4.7 API connection"""
    print("Testing GLM-4.7 API connection...")
    
    api_key = os.getenv("GLM_API_KEY", "78e909a9cf7b48a2856a1b178fbd4e7d.ZKmtkKseITStcyrE")
    client = ZhipuAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": "Say 'Hello GLM-4.7' in exactly 3 words."}],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        print(f"GLM-4.7 Response: {result}")
        print("[SUCCESS] GLM-4.7 connection working!")
        return True
        
    except Exception as e:
        print(f"[ERROR] GLM-4.7 connection failed: {e}")
        return False

def test_claim_decomposition():
    """Test claim decomposition with GLM-4.7"""
    print("\nTesting claim decomposition...")
    
    try:
        from src.claim_decomposer import decompose_claim
        
        test_claim = "John was a brave soldier who fought in the Battle of Waterloo."
        atoms = decompose_claim(test_claim)
        
        print(f"Original claim: {test_claim}")
        print(f"Decomposed atoms: {atoms}")
        print("[SUCCESS] Claim decomposition working!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Claim decomposition failed: {e}")
        return False

if __name__ == "__main__":
    print("GLM-4.7 Integration Test\n")
    
    success_count = 0
    
    if test_glm_connection():
        success_count += 1
    
    if test_claim_decomposition():
        success_count += 1
    
    print(f"\n[SUMMARY] {success_count}/2 tests passed")
    
    if success_count == 2:
        print("[SUCCESS] GLM-4.7 integration complete!")
    else:
        print("[WARNING] Some tests failed - check configuration")