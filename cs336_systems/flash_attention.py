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
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for k_tile_index in range(Tk):
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

# backward triton
@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    
    # Load K, V
    K_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
    V_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
    # Initialize accumulators
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    for q_tile_index in range (Tq):
        # Load Q, O, dO, L, D
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option='zero')
        O_tile = tl.load(O_block_ptr, boundary_check=(0,1), padding_option='zero')
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option='zero')
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
        
        # compute D
        D_tile = tl.sum(O_tile * dO_tile, axis=1) 
        
        # compute S, P
        S_tile = tl.dot(Q_tile.to(tl.float16), tl.trans(K_tile).to(tl.float16)) * scale
        if is_causal:
            # Generate relative indices and add offset to get global indices
            q_indices = tl.arange(0, Q_TILE_SIZE) + q_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
            mask = q_indices[:, None] >= k_indices[None, :]
            S_tile = tl.where(mask, S_tile, -1e6)
        P_tile = tl.exp(S_tile - L_tile[:, None])
        
        # compute dV, dP, dS
        dV_j += tl.dot(tl.trans(P_tile.to(tl.float16)), dO_tile.to(tl.float16))
        dP_tile = tl.dot(dO_tile.to(tl.float16), tl.trans(V_tile).to(tl.float16))
        dS_tile = P_tile * (dP_tile - D_tile[:, None]) * scale

        # atomic add dQ
        dQ_update = tl.dot(dS_tile.to(tl.float16), K_tile.to(tl.float16))
        # apply mask to handle boundary violation
        q_offsets = tl.arange(0, Q_TILE_SIZE) + q_tile_index * Q_TILE_SIZE
        d_offsets = tl.arange(0, D)  
        dQ_ptrs = dQ_ptr + batch_index * stride_dqb + q_offsets[:, None] * stride_dqq + d_offsets[None, :] * stride_dqd
        q_mask = (q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D)
        tl.atomic_add(dQ_ptrs, dQ_update, mask=q_mask)
        
        # accumulate dK
        dK_j += tl.dot(tl.trans(dS_tile.to(tl.float16)), Q_tile.to(tl.float16))
        
        # advance block ptrs
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, ))
        
    # store dK and dV
    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty), boundary_check=(0,1))
        
        
        
class Flash_Attention_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Save original input (for backward)
        Q_orig, K_orig, V_orig = Q, K, V

        # Handle 3D and 4D input
        if Q.dim() == 3:
            squeeze_out = True
            batch_size, seq_len_q, d = Q.shape
            n_heads = 1
        else:
            squeeze_out = False
            batch_size, n_heads, seq_len_q, d = Q.shape
            Q = einops.rearrange(Q, "b h s_q d -> (b h) s_q d")
            K = einops.rearrange(K, "b h s_k d -> (b h) s_k d")
            V = einops.rearrange(V, "b h s_k d -> (b h) s_k d")

        seq_len_k = K.shape[-2]

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

        # Reshape back to (batch, heads, seq, d) if input was 4D
        if not squeeze_out:
            O = einops.rearrange(O, "(b h) s_q d -> b h s_q d", h=n_heads)
            L = einops.rearrange(L, "(b h) s_q -> b h s_q", h=n_heads)

        # Save for backward
        ctx.save_for_backward(Q_orig, K_orig, V_orig, O, L)
        ctx.is_causal = is_causal

        return O
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        dO = grad_out
        # handle 3D input
        squeeze_out = False
        if Q.dim() == 3:
            squeeze_out = True
            batch_size, seq_len_q, d = Q.shape
            n_heads = 1
        else:
            squeeze_out = False
            batch_size, n_heads, seq_len_q, d = Q.shape
            Q = einops.rearrange(Q, "b h s d -> (b h) s d")
            K = einops.rearrange(K, "b h s d -> (b h) s d")
            V = einops.rearrange(V, "b h s d -> (b h) s d")
            O = einops.rearrange(O, "b h s d -> (b h) s d")
            L = einops.rearrange(L, "b h s -> (b h) s")
            dO = einops.rearrange(dO, "b h s d -> (b h) s d")
        
        seq_len_k = K.shape[-2]
        
        # Tile sizes
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        
        # scale
        scale = 1 / math.sqrt(d)
        
        # initialize dQ, dK, dV
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Get strides (assuming contiguous tensors)
        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vk, stride_vd = V.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq = L.stride()
        stride_dob, stride_doq, stride_dod = dO.stride()
        stride_dqb, stride_dqq, stride_dqd = dQ.stride()
        stride_dkb, stride_dkk, stride_dkd = dK.stride()
        stride_dvb, stride_dvk, stride_dvd = dV.stride()
        
        # Launch kernel
        Tk = cdiv(seq_len_k, K_TILE_SIZE)
        grid = (Tk, batch_size * n_heads)
        
       
        flash_bwd_kernel[grid](
            Q, K, V, O, L, dO, dQ, dK, dV,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            stride_dob, stride_doq, stride_dod,
            stride_dqb, stride_dqq, stride_dqd,
            stride_dkb, stride_dkk, stride_dkd,
            stride_dvb, stride_dvk, stride_dvd,
            seq_len_q, seq_len_k,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )
        
        # Reshape back to (b, h, s, d) if input was 4D
        if not squeeze_out:
            dQ = einops.rearrange(dQ, "(b h) s d -> b h s d", h=n_heads)
            dK = einops.rearrange(dK, "(b h) s d -> b h s d", h=n_heads)
            dV = einops.rearrange(dV, "(b h) s d -> b h s d", h=n_heads)

        return dQ, dK, dV, None