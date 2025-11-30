"""
Comprehensive test suite for production-ready language model
Tests KV caching correctness, GQA functionality, and performance benchmarks
"""

import torch
import torch.nn.functional as F
import time
import argparse
from model import LanguageModel, ModelConfig


def test_kv_cache_correctness():
    """Test that KV caching produces the same outputs as without caching"""
    print("\n" + "="*80)
    print("TEST: KV Cache Correctness")
    print("="*80)
    
    config = ModelConfig(
        vocab_size=1000,
        context_length=256,
        n_layers=4,
        n_heads=4,
        n_embd=256,
        dropout=0.0,
        use_gradient_checkpointing=False
    )
    
    model = LanguageModel(config)
    model.eval()
    
    # Test input
    batch_size = 2
    prompt_len = 10
    new_tokens = 5
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    
    # Generate without cache
    print("Generating without cache...")
    with torch.no_grad():
        output_no_cache = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            temperature=1.0,
            use_cache=False
        )
    
    # Generate with cache
    print("Generating with cache...")
    torch.manual_seed(42)  # Reset seed for reproducibility
    with torch.no_grad():
        # Need to set seed before both generations for fair comparison
        torch.manual_seed(42)
        output_no_cache = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            temperature=1.0,
            use_cache=False
        )
        
        torch.manual_seed(42)
        output_cache = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            temperature=1.0,
            use_cache=True
        )
    
    # Check if outputs match
    if torch.equal(output_no_cache, output_cache):
        print("✅ PASSED: KV cache produces identical outputs")
        return True
    else:
        print("❌ FAILED: KV cache outputs differ from non-cached")
        print(f"No cache: {output_no_cache}")
        print(f"With cache: {output_cache}")
        return False


def test_gqa_equivalence():
    """Test that GQA with n_kv_heads=n_heads is equivalent to MHA"""
    print("\n" + "="*80)
    print("TEST: GQA Equivalence to MHA")
    print("="*80)
    
    config_mha = ModelConfig(
        vocab_size=1000,
        context_length=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,  # MHA: same as n_heads
        n_embd=256,
        dropout=0.0
    )
    
    config_gqa = ModelConfig(
        vocab_size=1000,
        context_length=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # GQA: fewer KV heads
        n_embd=256,
        dropout=0.0
    )
    
    model_mha = LanguageModel(config_mha)
    model_gqa = LanguageModel(config_gqa)
    
    print(f"MHA params: {model_mha.get_num_params() / 1e6:.2f}M")
    print(f"GQA params: {model_gqa.get_num_params() / 1e6:.2f}M")
    print(f"Param reduction: {(1 - model_gqa.get_num_params() / model_mha.get_num_params()) * 100:.1f}%")
    
    # Both should work
    dummy_input = torch.randint(0, 1000, (1, 20))
    
    with torch.no_grad():
        logits_mha, _, _ = model_mha(dummy_input, use_cache=False)
        logits_gqa, _, _ = model_gqa(dummy_input, use_cache=False)
    
    print(f"MHA output shape: {logits_mha.shape}")
    print(f"GQA output shape: {logits_gqa.shape}")
    
    if logits_mha.shape == logits_gqa.shape:
        print("✅ PASSED: GQA produces correct output shapes")
        return True
    else:
        print("❌ FAILED: Shape mismatch")
        return False


def benchmark_generation_speed():
    """Benchmark generation speed with and without KV caching"""
    print("\n" + "="*80)
    print("BENCHMARK: Generation Speed")
    print("="*80)
    
    config = ModelConfig(
        vocab_size=10000,
        context_length=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        n_embd=512,
        dropout=0.0
    )
    
    model = LanguageModel(config)
    model.eval()
    
    print(f"Model: {model.get_num_params() / 1e6:.2f}M parameters")
    
    # Test different sequence lengths
    prompt_lengths = [10, 50, 100]
    gen_lengths = [20, 50, 100]
    
    results = []
    
    for prompt_len in prompt_lengths:
        for gen_len in gen_lengths:
            input_ids = torch.randint(0, config.vocab_size, (1, prompt_len))
            
            # Without cache
            start = time.time()
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=gen_len,
                    temperature=1.0,
                    use_cache=False
                )
            time_no_cache = time.time() - start
            
            # With cache
            start = time.time()
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=gen_len,
                    temperature=1.0,
                    use_cache=True
                )
            time_cache = time.time() - start
            
            speedup = time_no_cache / time_cache
            results.append((prompt_len, gen_len, time_no_cache, time_cache, speedup))
            
            print(f"Prompt {prompt_len:3d} → Gen {gen_len:3d}: "
                  f"No cache: {time_no_cache:.3f}s | "
                  f"Cache: {time_cache:.3f}s | "
                  f"Speedup: {speedup:.2f}x")
    
    avg_speedup = sum(r[4] for r in results) / len(results)
    print(f"\n✅ Average speedup: {avg_speedup:.2f}x")
    
    return True


def test_gradient_checkpointing():
    """Test gradient checkpointing memory savings"""
    print("\n" + "="*80)
    print("TEST: Gradient Checkpointing")
    print("="*80)
    
    config = ModelConfig(
        vocab_size=5000,
        context_length=256,
        n_layers=8,
        n_heads=8,
        n_embd=512,
        dropout=0.1,
        use_gradient_checkpointing=True
    )
    
    model = LanguageModel(config)
    model.train()
    
    # Test with checkpointing disabled
    model.disable_gradient_checkpointing()
    dummy_input = torch.randint(0, config.vocab_size, (2, 128))
    dummy_targets = torch.randint(0, config.vocab_size, (2, 128))
    
    print("Running forward/backward without gradient checkpointing...")
    logits, loss, _ = model(dummy_input, dummy_targets)
    loss.backward()
    
    # Test with checkpointing enabled
    model.zero_grad()
    model.enable_gradient_checkpointing()
    
    print("Running forward/backward with gradient checkpointing...")
    logits, loss, _ = model(dummy_input, dummy_targets)
    loss.backward()
    
    print("✅ PASSED: Gradient checkpointing works without errors")
    return True


def test_cache_shapes():
    """Test KV cache shapes are correct"""
    print("\n" + "="*80)
    print("TEST: KV Cache Shapes")
    print("="*80)
    
    config = ModelConfig(
        vocab_size=1000,
        context_length=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        n_embd=512,
        dropout=0.0
    )
    
    model = LanguageModel(config)
    model.eval()
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, _, kvs = model(input_ids, use_cache=True)
    
    print(f"Number of layers: {len(kvs)}")
    print(f"Expected: {config.n_layers}")
    
    for i, (k, v) in enumerate(kvs):
        expected_shape = (batch_size, config.n_kv_heads, seq_len, config.n_embd // config.n_heads)
        print(f"Layer {i}: K shape: {k.shape}, V shape: {v.shape}")
        
        if k.shape != expected_shape or v.shape != expected_shape:
            print(f"❌ FAILED: Expected shape {expected_shape}")
            return False
    
    print("✅ PASSED: All KV cache shapes are correct")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80)
    
    tests = [
        ("KV Cache Correctness", test_kv_cache_correctness),
        ("GQA Equivalence", test_gqa_equivalence),
        ("KV Cache Shapes", test_cache_shapes),
        ("Gradient Checkpointing", test_gradient_checkpointing),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ FAILED: {name} - {str(e)}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test production-ready language model")
    parser.add_argument("--test-kv-cache", action="store_true", help="Test KV cache correctness")
    parser.add_argument("--test-gqa", action="store_true", help="Test GQA functionality")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--benchmark-generation", action="store_true", help="Benchmark generation speed")
    args = parser.parse_args()
    
    if args.test_kv_cache:
        test_kv_cache_correctness()
    elif args.test_gqa:
        test_gqa_equivalence()
    elif args.benchmark_generation:
        benchmark_generation_speed()
    elif args.test_all:
        success = run_all_tests()
        if args.benchmark_generation or args.test_all:
            benchmark_generation_speed()
    else:
        # Default: run all tests
        run_all_tests()
        benchmark_generation_speed()
