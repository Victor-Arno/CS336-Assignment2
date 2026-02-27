from cs336_systems.flash_attention import Flash_Attention_Pytorch, Flash_Attention_Triton
from triton.testing import do_bench
import torch
import math
import einops
import pandas as pd


def pytorch_attention(Q, K, V, is_causal=True):
    d = Q.shape[-1]
    seq_len = Q.shape[-2]
    scale = 1 / math.sqrt(d)
    S = einops.einsum(Q, K, "... s_q d, ... s_k d -> ... s_q s_k") * scale
    if is_causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        S = S.masked_fill(~mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O


def benchmark_flash_attention():
    batch_size = 1
    is_causal = True
    device = 'cuda'

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_models = [16, 32, 64, 128]

    # bfloat16 requires sm_80+ (Ampere/H100); skip on older GPUs
    major, _ = torch.cuda.get_device_capability()
    dtypes = [torch.bfloat16, torch.float32] if major >= 8 else [torch.float32]
    if major < 8:
        print(f"Warning: GPU sm_{major}x detected, skipping bfloat16 (requires sm_80+)")

    results = []

    for dtype in dtypes:
        for d_model in d_models:
            for seq_len in seq_lens:
                row = {
                    'dtype': str(dtype).replace('torch.', ''),
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'fwd_pytorch(ms)': 'OOM',
                    'fwd_triton(ms)':  'OOM',
                    'bwd_pytorch(ms)': 'OOM',
                    'bwd_triton(ms)':  'OOM',
                    'e2e_pytorch(ms)': 'OOM',
                    'e2e_triton(ms)':  'OOM',
                }

                Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

                # --- Forward: PyTorch ---
                try:
                    fwd_pytorch = do_bench(lambda: pytorch_attention(Q, K, V, is_causal))
                    row['fwd_pytorch(ms)'] = f'{fwd_pytorch:.3f}'
                except torch.cuda.OutOfMemoryError:
                    fwd_pytorch = None
                    torch.cuda.empty_cache()

                # --- Forward: Triton ---
                try:
                    fwd_triton = do_bench(lambda: Flash_Attention_Triton.apply(Q, K, V, is_causal))
                    row['fwd_triton(ms)'] = f'{fwd_triton:.3f}'
                except torch.cuda.OutOfMemoryError:
                    fwd_triton = None
                    torch.cuda.empty_cache()

                # --- End-to-end: PyTorch (fwd + bwd) ---
                try:
                    def e2e_pytorch_fn():
                        out = pytorch_attention(Q, K, V, is_causal)
                        out.sum().backward()
                        Q.grad = K.grad = V.grad = None

                    e2e_pytorch = do_bench(e2e_pytorch_fn)
                    row['e2e_pytorch(ms)'] = f'{e2e_pytorch:.3f}'

                    if fwd_pytorch is not None:
                        row['bwd_pytorch(ms)'] = f'{e2e_pytorch - fwd_pytorch:.3f}'
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

                # --- End-to-end: Triton (fwd + bwd) ---
                try:
                    def e2e_triton_fn():
                        out = Flash_Attention_Triton.apply(Q, K, V, is_causal)
                        out.sum().backward()
                        Q.grad = K.grad = V.grad = None

                    e2e_triton = do_bench(e2e_triton_fn)
                    row['e2e_triton(ms)'] = f'{e2e_triton:.3f}'

                    if fwd_triton is not None:
                        row['bwd_triton(ms)'] = f'{e2e_triton - fwd_triton:.3f}'
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

                del Q, K, V
                torch.cuda.empty_cache()

                results.append(row)
                print(f"done: dtype={row['dtype']} d_model={d_model} seq_len={seq_len}")

    df = pd.DataFrame(results)
    print('\n' + df.to_string(index=False))


if __name__ == "__main__":
    benchmark_flash_attention()
