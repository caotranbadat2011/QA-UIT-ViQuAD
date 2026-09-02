"""
Quick test script to verify the model and tokenizer are working correctly.
Run this before starting the web app to ensure everything is set up properly.
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("[TEST] Testing package imports...")
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        
        import transformers
        print(f"  [OK] Transformers {transformers.__version__}")
        
        import streamlit
        print(f"  [OK] Streamlit {streamlit.__version__}")
        
        import pandas
        print(f"  [OK] Pandas {pandas.__version__}")
        
        import numpy
        print(f"  [OK] NumPy {numpy.__version__}")
        
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        print("\n[INFO] Please install dependencies: pip install -r requirements.txt")
        return False


def test_model_loading():
    """Test if model and tokenizer can be loaded"""
    print("\n[TEST] Testing model loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        
        model_dir = "models/phobert_qa"
        if not Path(model_dir).exists():
            print(f"  [FAIL] Model directory not found: {model_dir}")
            return False
        
        print(f"  [INFO] Loading from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"  [OK] Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        print(f"  [OK] Model loaded (parameters: {sum(p.numel() for p in model.parameters()):,})")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error loading model: {e}")
        return False


def test_inference():
    """Test a simple inference"""
    print("\n[TEST] Testing inference...")
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        import torch
        
        model_dir = "models/phobert_qa"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        model.eval()
        
        # Test input
        question = "Ha Noi o dau?"
        context = "Ha Noi la thu do cua Viet Nam, nam ben bo song Hong."
        
        inputs = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=256,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Get predicted answer
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        
        # Decode tokens
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        print(f"  [Q] Question: {question}")
        print(f"  [CTX] Context: {context}")
        print(f"  [ANS] Answer: {answer}")
        print(f"  [OK] Inference successful!")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Inference error: {e}")
        return False


def main():
    print("=" * 60)
    print("  Vietnamese QA System - Pre-deployment Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Package Imports", test_imports()))
    
    if not results[-1][1]:
        print("\n[FAIL] Package import test failed. Please install dependencies first.")
        sys.exit(1)
    
    # Test 2: Model Loading
    results.append(("Model Loading", test_model_loading()))
    
    if not results[-1][1]:
        print("\n[FAIL] Model loading test failed. Check model directory and files.")
        sys.exit(1)
    
    # Test 3: Inference
    results.append(("Inference", test_inference()))
    
    if not results[-1][1]:
        print("\n[FAIL] Inference test failed. Check model compatibility.")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! You're ready to run the web app.")
        print("\n[INFO] Run command: streamlit run app/app.py")
        print("   Or use: run_app.bat (Windows)")
    else:
        print("\n[WARNING] Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
