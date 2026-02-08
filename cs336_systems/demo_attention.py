from cs336_basics.model import scaled_dot_product_attention
import einops
import math
from cs336_basics.nn_utils import softmax
import torch
import timeit


def test_attention_impl(attention_fn, Q, K, V, warmup_steps, test_steps):
    """Test one attention implementation and return timings and memory"""

    # Warm-up
    for _ in range(warmup_steps):
        with torch.no_grad():
            output = attention_fn(Q, K, V)
        torch.cuda.synchronize()

    # Time forward passes
    torch.cuda.synchronize()
    start = timeit.default_timer()
    for _ in range(test_steps):
        with torch.no_grad():
            output = attention_fn(Q, K, V)
        torch.cuda.synchronize()
    forward_time = (timeit.default_timer() - start) * 1000  # ms

    # Measure memory before backward
    output = attention_fn(Q, K, V)
    torch.cuda.synchronize()
    memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

    # Clean up computation graph
    output.sum().backward()
    Q.grad = None
    K.grad = None
    V.grad = None

    # Time backward passes
    torch.cuda.synchronize()
    start = timeit.default_timer()
    for _ in range(test_steps):
        output = attention_fn(Q, K, V)
        output.sum().backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        torch.cuda.synchronize()
    backward_time = (timeit.default_timer() - start) * 1000  # ms

    return forward_time, backward_time, memory


def benchmark_attention():
    batch_size = 8
    warmup_steps = 10
    test_steps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    # Prepare both versions
    original_attn = scaled_dot_product_attention
    compiled_attn = torch.compile(scaled_dot_product_attention)

    print(f"Device: {device}")
    print(f"{'d_model':<10} {'seq_len':<10} {'fwd_orig(ms)':<17} {'bwd_orig(ms)':<17} {'mem_orig(MB)':<17} {'fwd_comp(ms)':<17} {'bwd_comp(ms)':<17} {'mem_comp(MB)':<17}")
    print("-" * 130)

    for d_model in d_models:
        for seq_len in seq_lens:
            try:
                # Create Q, K, V once
                Q = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
                K = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
                V = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)

                # Test original version
                fwd_orig, bwd_orig, mem_orig = test_attention_impl(
                    original_attn, Q, K, V, warmup_steps, test_steps
                )

                # Clean up and recreate tensors for compiled version
                del Q, K, V
                torch.cuda.empty_cache()

                Q = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
                K = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
                V = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)

                # Test compiled version
                fwd_comp, bwd_comp, mem_comp = test_attention_impl(
                    compiled_attn, Q, K, V, warmup_steps, test_steps
                )

                # Print results
                print(f"{d_model:<10} {seq_len:<10} {fwd_orig:<17.2f} {bwd_orig:<17.2f} {mem_orig:<17.2f} {fwd_comp:<17.2f} {bwd_comp:<17.2f} {mem_comp:<17.2f}")

                # Clean up
                del Q, K, V
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"{d_model:<10} {seq_len:<10} {'OOM':<17} {'OOM':<17} {'OOM':<17} {'OOM':<17} {'OOM':<17} {'OOM':<17}")
                torch.cuda.empty_cache()
                continue


if __name__ == "__main__":
    benchmark_attention()
