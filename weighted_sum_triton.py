"""
完整的 Triton Weighted Sum 实现
包括 forward kernel、backward kernel 和 PyTorch wrapper
"""

import torch
import triton
import triton.language as tl
from einops import rearrange


def cdiv(a, b):
    """向上取整除法"""
    return (a + b - 1) // b


# ==================== Forward Kernel ====================

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,      # 输入指针
    output_ptr,             # 输出指针
    x_stride_row, x_stride_dim,  # x 的步长
    weight_stride_dim,      # weight 的步长
    output_stride_row,      # output 的步长
    ROWS, D,                # 矩阵维度
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # 分块大小（编译时常量）
):
    """
    计算加权和: output[i] = sum_j(x[i, j] * weight[j])

    每个线程块处理 ROWS_TILE_SIZE 行
    循环处理 D 维度，每次处理 D_TILE_SIZE 列
    """
    # 获取当前线程块的索引
    row_tile_idx = tl.program_id(0)

    # 创建 x 的 block pointer
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),  # 每个线程块处理不同的行
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),  # 列主序
    )

    # 创建 weight 的 block pointer
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # 创建 output 的 block pointer
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 初始化输出缓冲区（在寄存器中）
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # 循环处理 D 维度
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载 x 的一个块: (ROWS_TILE_SIZE, D_TILE_SIZE)
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # 加载 weight 的一个块: (D_TILE_SIZE,)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        # 计算加权和并累加
        # row: (ROWS_TILE_SIZE, D_TILE_SIZE)
        # weight[None, :]: (1, D_TILE_SIZE)
        # 结果: (ROWS_TILE_SIZE, D_TILE_SIZE) -> sum(axis=1) -> (ROWS_TILE_SIZE,)
        output += tl.sum(row * weight[None, :], axis=1)

        # 移动指针到下一个块
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    # 写回结果到全局内存
    tl.store(output_block_ptr, output, boundary_check=(0,))


# ==================== Backward Kernel ====================

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,      # 输入（forward 的输入）
    grad_output_ptr,        # 输入（对 output 的梯度）
    grad_x_ptr, partial_grad_weight_ptr,  # 输出（对 x 和 weight 的梯度）
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    """
    计算梯度:
    - grad_x[i, j] = grad_output[i] * weight[j]  (外积)
    - grad_weight[j] = sum_i(grad_output[i] * x[i, j])  (需要跨行求和)

    每个线程块计算 grad_weight 的部分和（只对自己的行）
    """
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)  # 线程块总数

    # === 创建 block pointers ===

    # grad_output: (NUM_ROWS,) - 对输出的梯度
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # x: (NUM_ROWS, D) - forward 的输入
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # weight: (D,) - forward 的输入
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # grad_x: (NUM_ROWS, D) - 对 x 的梯度（输出）
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # partial_grad_weight: (n_row_tiles, D) - 对 weight 的部分梯度（输出）
    # 每个线程块写一行，后续需要对所有行求和
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),  # 注意：这里是 row_tile_idx，不是 row_tile_idx * ROWS_TILE_SIZE
        block_shape=(1, D_TILE_SIZE),  # 每次写 1 行
        order=(1, 0),
    )

    # === 主循环 ===

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载 grad_output: (ROWS_TILE_SIZE,)
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

        # === 计算 grad_x (方程 2: 外积) ===
        # grad_x[i, j] = grad_output[i] * weight[j]
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_x_tile = grad_output[:, None] * weight[None, :]  # (ROWS_TILE_SIZE, 1) * (1, D_TILE_SIZE)
        tl.store(grad_x_block_ptr, grad_x_tile, boundary_check=(0, 1))

        # === 计算 partial_grad_weight (方程 3: 部分和) ===
        # grad_weight[j] = sum_i(grad_output[i] * x[i, j])
        # 这里只计算当前线程块负责的行的部分和
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        partial_grad_weight_tile = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        # (ROWS_TILE_SIZE, D_TILE_SIZE) -> sum(axis=0) -> (1, D_TILE_SIZE)
        tl.store(partial_grad_weight_block_ptr, partial_grad_weight_tile, boundary_check=(1,))

        # 移动指针到下一个块
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


# ==================== PyTorch Autograd Function ====================

class WeightedSumFunc(torch.autograd.Function):
    """
    PyTorch autograd 函数，包装 Triton kernel
    使用方式: output = WeightedSumFunc.apply(x, weight)
    """

    @staticmethod
    def forward(ctx, x, weight):
        """
        前向传播

        Args:
            x: (batch, seq_len, D) 或任意形状 (..., D)
            weight: (D,)

        Returns:
            output: (...,) 去掉最后一维
        """
        # 保存输入形状
        input_shape = x.shape
        D = x.shape[-1]
        output_dims = x.shape[:-1]

        # 将 x 展平成 2D: (..., D) -> (n_rows, D)
        x = rearrange(x, "... d -> (...) d")
        n_rows = x.shape[0]

        # 保存用于 backward
        ctx.save_for_backward(x, weight)

        # 断言
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "x must be contiguous"

        # 设置分块大小
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # 大约 16 次循环
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        # 分配输出张量
        y = torch.empty(n_rows, device=x.device, dtype=x.dtype)

        # 启动 kernel
        grid = (cdiv(n_rows, ctx.ROWS_TILE_SIZE),)
        weighted_sum_fwd[grid](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        # 恢复原始形状（去掉最后一维）
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播

        Args:
            grad_out: (...,) 对 output 的梯度

        Returns:
            grad_x: (..., D) 对 x 的梯度
            grad_weight: (D,) 对 weight 的梯度
        """
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE
        D_TILE_SIZE = ctx.D_TILE_SIZE
        input_shape = ctx.input_shape

        n_rows, D = x.shape

        # grad_out 也需要展平
        grad_out = grad_out.contiguous().view(-1)

        # 分配输出张量
        grad_x = torch.empty_like(x)

        # partial_grad_weight: 每个线程块一行
        n_row_tiles = cdiv(n_rows, ROWS_TILE_SIZE)
        partial_grad_weight = torch.empty((n_row_tiles, D), device=x.device, dtype=x.dtype)

        # 启动 backward kernel
        grid = (n_row_tiles,)
        weighted_sum_backward[grid](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        # 第二阶段：对 partial_grad_weight 求和
        grad_weight = partial_grad_weight.sum(axis=0)

        # 恢复 grad_x 的原始形状
        grad_x = grad_x.view(input_shape)

        return grad_x, grad_weight


# 创建可调用函数
f_weightedsum = WeightedSumFunc.apply


# ==================== 测试和验证 ====================

def test_forward():
    """测试 forward pass 的正确性"""
    print("=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    # 创建测试数据
    batch, seq_len, D = 32, 128, 512
    x = torch.randn(batch, seq_len, D, device='cuda', requires_grad=True)
    weight = torch.randn(D, device='cuda', requires_grad=True)

    # Triton 实现
    output_triton = f_weightedsum(x, weight)

    # PyTorch 原生实现
    output_pytorch = (x * weight).sum(-1)

    # 比较结果
    max_diff = (output_triton - output_pytorch).abs().max().item()
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {output_triton.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"All close: {torch.allclose(output_triton, output_pytorch, atol=1e-5)}")
    print()


def test_backward():
    """测试 backward pass 的正确性"""
    print("=" * 60)
    print("Testing Backward Pass")
    print("=" * 60)

    # 创建测试数据
    batch, seq_len, D = 32, 128, 512
    x = torch.randn(batch, seq_len, D, device='cuda', requires_grad=True)
    weight = torch.randn(D, device='cuda', requires_grad=True)

    # === Triton 实现 ===
    output_triton = f_weightedsum(x, weight)
    loss_triton = output_triton.sum()
    loss_triton.backward()
    grad_x_triton = x.grad.clone()
    grad_weight_triton = weight.grad.clone()

    # 清空梯度
    x.grad = None
    weight.grad = None

    # === PyTorch 原生实现 ===
    output_pytorch = (x * weight).sum(-1)
    loss_pytorch = output_pytorch.sum()
    loss_pytorch.backward()
    grad_x_pytorch = x.grad.clone()
    grad_weight_pytorch = weight.grad.clone()

    # 比较结果
    max_diff_x = (grad_x_triton - grad_x_pytorch).abs().max().item()
    max_diff_weight = (grad_weight_triton - grad_weight_pytorch).abs().max().item()

    print(f"grad_x max difference: {max_diff_x:.2e}")
    print(f"grad_x all close: {torch.allclose(grad_x_triton, grad_x_pytorch, atol=1e-5)}")
    print(f"grad_weight max difference: {max_diff_weight:.2e}")
    print(f"grad_weight all close: {torch.allclose(grad_weight_triton, grad_weight_pytorch, atol=1e-5)}")
    print()


def test_grad_fn():
    """测试 grad_fn 是否正确附加"""
    print("=" * 60)
    print("Testing grad_fn")
    print("=" * 60)

    x = torch.randn(10, 512, device='cuda', requires_grad=True)
    weight = torch.randn(512, device='cuda', requires_grad=True)

    output = f_weightedsum(x, weight)

    print(f"Output sample: {output[:5]}")
    print(f"requires_grad: {output.requires_grad}")
    print(f"grad_fn: {output.grad_fn}")
    print(f"grad_fn class: {output.grad_fn.__class__.__name__}")
    print()


def test_different_shapes():
    """测试不同输入形状"""
    print("=" * 60)
    print("Testing Different Shapes")
    print("=" * 60)

    test_cases = [
        ((1000, 512), "2D input"),
        ((32, 64, 512), "3D input (batch, seq, dim)"),
        ((8, 16, 32, 512), "4D input"),
    ]

    for shape, description in test_cases:
        D = shape[-1]
        x = torch.randn(*shape, device='cuda', requires_grad=True)
        weight = torch.randn(D, device='cuda', requires_grad=True)

        output_triton = f_weightedsum(x, weight)
        output_pytorch = (x * weight).sum(-1)

        is_close = torch.allclose(output_triton, output_pytorch, atol=1e-5)
        print(f"{description}: {shape} -> {output_triton.shape}, correct: {is_close}")

    print()


def benchmark():
    """简单的性能测试"""
    print("=" * 60)
    print("Benchmarking")
    print("=" * 60)

    import timeit

    batch, seq_len, D = 32, 1024, 512
    x = torch.randn(batch, seq_len, D, device='cuda')
    weight = torch.randn(D, device='cuda')

    # 预热
    for _ in range(10):
        _ = f_weightedsum(x, weight)
        _ = (x * weight).sum(-1)
    torch.cuda.synchronize()

    # Triton
    start = timeit.default_timer()
    for _ in range(100):
        output = f_weightedsum(x, weight)
    torch.cuda.synchronize()
    time_triton = (timeit.default_timer() - start) * 1000 / 100

    # PyTorch
    start = timeit.default_timer()
    for _ in range(100):
        output = (x * weight).sum(-1)
    torch.cuda.synchronize()
    time_pytorch = (timeit.default_timer() - start) * 1000 / 100

    print(f"Input shape: {x.shape}")
    print(f"Triton time: {time_triton:.3f} ms")
    print(f"PyTorch time: {time_pytorch:.3f} ms")
    print(f"Speedup: {time_pytorch / time_triton:.2f}x")
    print()


if __name__ == "__main__":
    # 运行所有测试
    test_forward()
    test_backward()
    test_grad_fn()
    test_different_shapes()
    benchmark()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
