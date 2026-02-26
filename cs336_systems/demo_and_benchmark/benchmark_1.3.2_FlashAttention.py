from cs336_systems.flash_attention import Flash_Attention_Pytorch, Flash_Attention_Triton
import torch
import triton.testing import do_bench
import math
import einops
import pandas as pd
def pytorch_attention(Q, K , V, is_causal=True):
    d = Q.shape[-1]
    seq_len = Q.shape[-2]
    scale = 1 / math.sqrt(d)
    S = einops.einsum(Q, K, " ... s_q d, ... s_k d -> ... s_q s_k") * scale
    # apply causal mask
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
    dtypes = [torch.bfloat16, torch.float32]
    
    # TODO: main loop
    
    # print the table of results(rather than csv file)
    pass

if __name__ == "__main__":
    benchmark_flash_attention()