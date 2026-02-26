import torch
import math
import einops
import triton.language as tl
import triton

# Ceiling division
def cdiv(a, b):
    return (a + b - 1) // b

# ==================== Part (a): PyTorch version ====================

@torch.compile
def compute_gradients(Q, K, V, O, L, dO, is_causal):
    # TODO
    # handle 3D input
    squeeze_out = False
    if Q.dim() == 3:
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)
        O = O.unsqueeze(1)
        L = L.unsqueeze(1)
        dO = dO.unsqueeze(1)
        squeeze_out = True
    
    # shape parameters
    batch_size, n_heads, seq_len_q, d = Q.shape
    seq_len_k = K.shape[-2]
    # scale
    scale = 1 / math.sqrt(d)
    # pre
    D = torch.sum(O * dO, dim=-1) # D: (s_q, )
    # recompute S
    S = einops.einsum(Q, K, "b h s_q d, b h s_k d -> b h s_q s_k") * scale
    # apply causal mask
    if is_causal:
        mask = torch.tril(torch.ones(seq_len_q,seq_len_k,device=Q.device)).bool()
        S = S.masked_fill(mask==False, -1e6)
        
    # recompute P
    P = torch.exp(S - L.unsqueeze(-1))  
    
    #  compute gradient
    dV = einops.einsum(P, dO, "... s_q s_k, ... s_q d -> ... s_k d")  
    dP = einops.einsum(dO, V, "... s_q d, ... s_k d -> ... s_q s_k")  
    dS = P * (dP - D.unsqueeze(-1))  
    dQ = einops.einsum(dS, K, "... s_q s_k, ... s_k d -> ... s_q d") * scale 
    dK = einops.einsum(dS, Q, "... s_q s_k, ... s_q d -> ... s_k d") * scale 
    
    if squeeze_out:
        dQ = dQ.squeeze(1)
        dK = dK.squeeze(1)
        dV = dV.squeeze(1)
        
    return dQ, dK, dV

class Flash_Attention_Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get shape parameters

        # Save original input (for backward)
        Q_orig, K_orig, V_orig = Q, K, V
        # Handle 3D input
        if Q.dim() == 3:
            Q = Q.unsqueeze(1)  # (batch, seq, d) -> (batch, 1, seq, d)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, n_heads, seq_len_q, d = Q.shape
        seq_len_k = K.shape[-2]

        # Set tile sizes
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        # Calculate number of tiles needed
        Tq = cdiv(seq_len_q, Q_TILE_SIZE)
        Tk = cdiv(seq_len_k, K_TILE_SIZE)

        # Initialize output tensors
        O = torch.zeros((batch_size, n_heads, seq_len_q, d), device=Q.device, dtype=Q.dtype)
        L = torch.zeros((batch_size, n_heads, seq_len_q), device=Q.device, dtype=torch.float32)

        # Calculate scale
        scale = 1 / math.sqrt(d)

        # Main loop
        # Outer loop: iterate over Q tiles
        for i in range(Tq):
            # Calculate q tile range
            q_start = Q_TILE_SIZE * i
            q_end = Q_TILE_SIZE * (i+1)

            # Extract Q_tile
            Q_tile = Q[:, :, q_start:q_end, :]

            # Initialize accumulators for this Q_tile
            O_i = torch.zeros((batch_size, n_heads, Q_TILE_SIZE, d), device=Q.device, dtype=Q.dtype)
            m_i = torch.full(
                (batch_size, n_heads, Q_TILE_SIZE),
                fill_value=-float('inf'),
                dtype=torch.float32,
                device=Q.device
            )
            l_i = torch.zeros((batch_size, n_heads, Q_TILE_SIZE), device=Q.device, dtype=torch.float32)

            # Inner loop: iterate over K tiles
            for j in range(Tk):
                # Initialize K_tile and V_tile
                k_start = K_TILE_SIZE * j
                k_end = K_TILE_SIZE * (j+1)
                K_tile = K[:, : , k_start:k_end, :]
                V_tile = V[:, : , k_start:k_end, :]

                # Compute S_i
                S_i = einops.einsum(Q_tile, K_tile, "... s_q d, ... s_k d -> ... s_q s_k") * scale

                # Apply causal mask if needed
                if is_causal:
                    # Global position indices
                    q_indices = torch.arange(q_start, q_end, device=Q.device)
                    k_indices = torch.arange(k_start, k_end, device=K.device)
                    # Construct causal mask
                    mask = q_indices[:, None] >= k_indices[None, :]
                    # Apply mask
                    S_i = S_i.masked_fill(mask == False, -1e6)

                # Record old values of m, l, O
                m_old = m_i
                l_old = l_i
                O_old = O_i
                # Compute new m
                m_new = torch.maximum(m_i, S_i.max(dim=-1).values)
                P_tiled = torch.exp(S_i - m_new.unsqueeze(-1))
                # Compute rescale_factor
                rescale_factor = torch.exp(m_old - m_new)
                # Compute new l
                l_new = rescale_factor * l_old + torch.sum(P_tiled, dim=-1)
                # Compute new O_i
                O_new = rescale_factor.unsqueeze(-1) * O_i + P_tiled @ V_tile
                # Update m_i, l_i, O_i
                m_i = m_new
                l_i = l_new
                O_i = O_new
            # Normalization
            # O_i_final
            O_i_final = O_i / l_i.unsqueeze(-1)
            # L_i_final
            L_i_final = m_i + torch.log(l_i)

            # Write back to output tensors
            O[:, :, q_start:q_end, :] = O_i_final
            L[:, :, q_start:q_end] = L_i_final


        # Squeeze back to 3D if needed
        if squeeze_output:
            O = O.squeeze(1)  # (batch, 1, seq, d) -> (batch, seq, d)
            L = L.squeeze(1)  # (batch, 1, seq) -> (batch, seq)

        # Save for backward
        ctx.save_for_backward(Q_orig, K_orig, V_orig, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx,grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = compute_gradients(Q, K, V, O, L, grad_out, ctx.is_causal)
        return dQ, dK, dV, None
        

# ==================== Part (b): Triton version ====================
# Triton kernel
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Get program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Define block pointers for Q, K, V, O, L
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    K_block_ptr = tl.make_block_ptr(
      K_ptr + batch_index * stride_kb,
      shape =(N_KEYS, D),
      strides=(stride_kk, stride_kd),
      offsets=(0, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1,0),
    )
    V_block_ptr = tl.make_block_ptr(
      V_ptr + batch_index * stride_vb,
      shape =(N_KEYS, D),
      strides=(stride_vk, stride_vd),
      offsets=(0, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1,0),
    )
    O_block_ptr = tl.make_block_ptr(
      O_ptr + batch_index * stride_ob,
      shape =(N_QUERIES, D),
      strides=(stride_oq, stride_od),
      offsets=(query_tile_index * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
      L_ptr + batch_index * stride_lb,
      shape =(N_QUERIES, ),
      strides=(stride_lq, ),
      offsets=(query_tile_index * Q_TILE_SIZE, ),
      block_shape=(Q_TILE_SIZE, ),
      order=(0, ),
    )

    # Initialize accumulators
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE, ), value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)

    # Load Q_tile
    Q_tile = tl.load(
        Q_block_ptr,
        boundary_check=(0,1),
        padding_option='zero',
    )
    for k_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load K_tile and V_tile
        K_tile = tl.load(
            K_block_ptr,
            boundary_check=(0,1),
            padding_option='zero',
        )
        V_tile = tl.load(
            V_block_ptr,
            boundary_check=(0,1),
            padding_option='zero',
        )
        S_tile = tl.dot(Q_tile.to(tl.float16), tl.trans(K_tile).to(tl.float16)) * scale

        # Apply causal mask if needed
        if is_causal:
            # Generate relative indices and add offset to get global indices
            q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE) + k_tile_index * K_TILE_SIZE
            mask = q_indices[:, None] >= k_indices[None, :]
            S_tile = tl.where(mask, S_tile, -1e6)

        # Record old accumulators
        m_old = m_i
        l_old = l_i

        # FlashAttention-2 online softmax algorithm
        m_new = tl.maximum(m_i, tl.max(S_tile, axis=1))
        P_tilde = tl.exp(S_tile - m_new[:,None])
        P_tilde = P_tilde.to(tl.float16)

        rescale_factor = tl.exp(m_old - m_new)
        l_new = rescale_factor * l_old + tl.sum(P_tilde, axis=1)

        # Update O_i, m_i, l_i
        O_i = rescale_factor[:, None] * O_i
        O_i = tl.dot(P_tilde, V_tile.to(tl.float16), acc=O_i)
        m_i = m_new
        l_i = l_new

        # Advance K and V block pointers to next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    # Write results back to HBM
    tl.store(
        O_block_ptr,
        (O_i / l_i[:, None]).to(O_block_ptr.type.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        L_block_ptr,
        m_i + tl.log(l_i),
        boundary_check=(0, ),
    )


class Flash_Attention_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Save original input (for backward)
        Q_orig, K_orig, V_orig = Q, K, V

        # Handle 3D and 4D input
        squeeze_out = False
        if Q.dim() == 3:
            Q = Q[:, None, :, :]
            K = K[:, None, :, :]
            V = V[:, None, :, :]
            squeeze_out = True

        batch_size, n_heads, seq_len_q, d = Q.shape
        seq_len_k = K.shape[-2]

        # Reshape to fit the Triton kernel (batch*heads, seq, d)
        Q = einops.rearrange(Q, "b h s_q d -> (b h) s_q d")
        K = einops.rearrange(K, "b h s_k d -> (b h) s_k d")
        V = einops.rearrange(V, "b h s_k d -> (b h) s_k d")

        # Initialize output tensors
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size * n_heads, seq_len_q), device=Q.device, dtype=torch.float32)

        # Tile sizes
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        # Scale factor
        scale = 1 / math.sqrt(d)

        # Get strides (assuming contiguous tensors)
        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vk, stride_vd = V.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq = L.stride()

        # Launch kernel
        Tq = cdiv(seq_len_q, Q_TILE_SIZE)
        grid = (Tq, batch_size * n_heads)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            seq_len_q, seq_len_k,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # Reshape back to (batch, heads, seq, d)
        O = einops.rearrange(O, "(b h) s_q d-> b h s_q d", h = n_heads)
        L = einops.rearrange(L, "(b h) s_q -> b h s_q ", h = n_heads)

        # Squeeze back to 3D if input was 3D
        if squeeze_out == True:
            O = O.squeeze(1)
            L = L.squeeze(1)

        # Save for backward
        ctx.save_for_backward(Q_orig, K_orig, V_orig, O, L)
        ctx.is_causal = is_causal

        return O
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented yet")
