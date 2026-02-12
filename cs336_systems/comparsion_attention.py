import torch
from cs336_systems.flash_attention import Flash_Attention_Pytorch
from cs336_systems.flash_attention import Flash_Attention_Triton

def test_basic():
    batch = 2
    heads = 4
    seq_len = 64
    d = 64
    
    Q = torch.randn(batch, heads, seq_len, d, device='cuda')
    K = torch.randn(batch, heads, seq_len, d, device='cuda')
    V = torch.randn(batch, heads, seq_len, d, device='cuda')
    
    # no causal
    print('-'*80)
    print("Without causal mask")
    O = Flash_Attention.apply(Q, K, V, False)
    print(f"Output shape: {O.shape}")
    print(f"Output sample: {O[0, 0, 0, :5]}")
    
    # with causal
    print('-'*80)
    print("With causal mask")
    O_causal = Flash_Attention.apply(Q, K, V, True)
    print(f"Causal output shape: {O_causal.shape}")
    print(f"Causal output sample: {O_causal[0, 0, 0, :5]}")
    
    print("Basic test passed!")
    
def test_correctness():
      """compare FlashAttention and standard attention"""
      batch = 2
      heads = 4
      seq_len = 64
      d = 64

      Q = torch.randn(batch, heads, seq_len, d, device='cuda', dtype=torch.float32)
      K = torch.randn(batch, heads, seq_len, d, device='cuda', dtype=torch.float32)
      V = torch.randn(batch, heads, seq_len, d, device='cuda', dtype=torch.float32)

      scale = 1 / (d ** 0.5)

      # standard PyTorch attention (without causal)
      S = torch.einsum('bhqd,bhkd->bhqk', Q, K) * scale
      P = torch.softmax(S, dim=-1)
      O_ref = torch.einsum('bhqk,bhkd->bhqd', P, V)

      # FlashAttention (without causal)
      O_flash = Flash_Attention.apply(Q, K, V, False)

      #
      max_diff = (O_ref - O_flash).abs().max().item()
      mean_diff = (O_ref - O_flash).abs().mean().item()

      print("=" * 80)
      print("Testing correctness (without causal)")
      print("=" * 80)
      print(f"Max difference: {max_diff:.6f}")
      print(f"Mean difference: {mean_diff:.6f}")
      print(f"All close (atol=1e-4): {torch.allclose(O_ref, O_flash, atol=1e-4)}")
      print()

      # test causal mask
      S_causal = torch.einsum('bhqd,bhkd->bhqk', Q, K) * scale
      mask = torch.tril(torch.ones(seq_len, seq_len, device='cuda')).bool()
      S_causal = S_causal.masked_fill(~mask, -1e6)
      P_causal = torch.softmax(S_causal, dim=-1)
      O_ref_causal = torch.einsum('bhqk,bhkd->bhqd', P_causal, V)

      O_flash_causal = Flash_Attention.apply(Q, K, V, True)

      max_diff_causal = (O_ref_causal - O_flash_causal).abs().max().item()
      mean_diff_causal = (O_ref_causal - O_flash_causal).abs().mean().item()

      print("=" * 80)
      print("Testing correctness (with causal)")
      print("=" * 80)
      print(f"Max difference: {max_diff_causal:.6f}")
      print(f"Mean difference: {mean_diff_causal:.6f}")
      print(f"All close (atol=1e-4): {torch.allclose(O_ref_causal, O_flash_causal,
  atol=1e-4)}")
      print()

      if torch.allclose(O_ref, O_flash, atol=1e-4) and torch.allclose(O_ref_causal,
  O_flash_causal, atol=1e-4):
          print(" All tests passed!")
      else:
          print("? Tests failed!")
          
if __name__ == "__main__":
    # test basis
    #    test_basic()
    
    # compare with pytorch
    test_correctness()