import torch
import torch.cuda.nvtx as nvtx
from cs336_systems.flash_attention import Flash_Attention

batch = 2
heads = 4
seq_len = 256
d = 64

Q = torch.randn(batch, heads, seq_len, d, device='cuda', dtype=torch.float32)
K = torch.randn(batch, heads, seq_len, d, device='cuda', dtype=torch.float32)
V = torch.randn(batch, heads, seq_len, d, device='cuda', dtype=torch.float32)

# Warm-up
for _ in range(5):
    O = Flash_Attention.apply(Q, K, V, False)
torch.cuda.synchronize()

# Profile with NVTX markers
with nvtx.range("FlashAttention_PyTorch"):
    for _ in range(10):
        O = Flash_Attention.apply(Q, K, V, False)
    torch.cuda.synchronize()

print(" Profiling completed!")
print(f"Output shape: {O.shape}")