from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW
import cs336_basics.model
import argparse
import torch
import timeit
import numpy as np
import torch.cuda.nvtx as nvtx
from einops import rearrange, einsum
from jaxtyping import Float, Bool, Int
from torch import Tensor
import math
from contextlib import nullcontext

# 命令行参数初始化
def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Benchmarking")
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--d_model", type=int, default=512, help="模型的维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=16, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=1344, help="FFN层的隐藏层维度")
    parser.add_argument("--context_length", type=int, default=256, help="上下文长度")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE的theta参数")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--test_steps", type=int, default=10, help="测量次数")
    parser.add_argument("--warmup_steps", type=int, default=5, help="预热次数")
    parser.add_argument("--mode", type=int, default=1, help="模式是否加入backward(0代表只有forward)")
    parser.add_argument("--mixed_precision", action="store_true", help="是否启用混合精度训练")
    parser.add_argument("--profile_memory", action="store_true", help="是否启用内存记录")
    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--betas", default=(0.9,0.999), help="betas")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="衰减权重")
    return parser.parse_args()

# 生成随机数据
def generate_randtext(batch_size,context_length,vocab_size):
    input_data  = torch.randint(0,vocab_size,(batch_size,context_length))
    return input_data

# benchmark函数
def benchmark(model,optimizer,input_data,warmup_steps,test_steps,mode,use_mixed_precision,profile_memory):
    # 是否采用混合精度训练
    ctx = torch.amp.autocast('cuda', dtype=torch.float16) if use_mixed_precision else nullcontext()
    # 1. warm-up预热(不计时,不记录内存)
    with nvtx.range("warm-up"):
      for i in range(warmup_steps):
          with ctx:
              output = model(input_data)
              if mode != 0:
                  output.sum().backward()
          if torch.cuda.is_available():
              torch.cuda.synchronize()
    # 是否记录内存
    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    # 2.测量计时
    with nvtx.range("benchmark"):
      times = []
      for i in range(test_steps):
          with ctx:
              start = timeit.default_timer()
              with nvtx.range("forward"):
                  output = model(input_data)
              with nvtx.range("backward"):
                if mode != 0:
                    output.sum().backward()
          with nvtx.range("optimizer_step"):
            if mode != 0:
                optimizer.step()
                optimizer.zero_grad()
          if torch.cuda.is_available():
              torch.cuda.synchronize()
          end = timeit.default_timer()
          times.append(end-start)
    if profile_memory:
        # 保存快照
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        # 停止记录
        torch.cuda.memory._record_memory_history(enabled=None)
    return times
      
# 替换的dot_product
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention score"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
        
    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def main():
    # 1. 解析命令行参数
    args = parse_args()
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 3. 初始化模型和优化器
    model = BasicsTransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.theta,
    )
    model = model.to(device)
    optimizer = AdamW(
        params=model.parameters(),
        lr = args.lr,
        betas = args.betas,
        eps = args.eps,
        weight_decay = args.weight_decay,
    )
    # 4. 生成随机数据
    input_data = generate_randtext(
        batch_size = args.batch_size,
        context_length = args.context_length,
        vocab_size = args.vocab_size
    )
    input_data = input_data.to(device)
    # 5. 替换scaled
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    # 6. 运行benchmark
    times = benchmark(
        model=model,
        optimizer=optimizer,
        input_data=input_data,
        warmup_steps=args.warmup_steps,
        test_steps=args.test_steps,
        mode=args.mode,
        use_mixed_precision=args.mixed_precision,
        profile_memory=args.profile_memory
    )
    # 7. 计算并打印结果（平均值和标准差）
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"平均时间: {mean_time:.4f}s, 标准差: {std_time:.4f}s")

if __name__ == "__main__":
    main()