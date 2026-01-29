from cs336_basics.model import BasicsTransformerLM
import cs336_basics.nn_utils as F
import argparse
import torch
import timeit
import numpy as np

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
    parser.add_argument("--mode", type=int, default=0, help="模式是否加入backward(0代表只有forward)")
    
    return parser.parse_args()

# 生成随机数据
def generate_randtext(batch_size,context_length,vocab_size):
    input_data  = torch.randint(0,vocab_size,(batch_size,context_length))
    return input_data

# benchmark函数
def benchmark(model,input_data,warmup_steps,test_steps,mode):
    # 1. warm-up预热(不计时)
    for i in range(warmup_steps):
        output = model(input_data)
        if mode != 0:
            output.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 2.测量计时
    times = []
    for i in range(test_steps):
        start = timeit.default_timer()
        output = model(input_data)
        if mode != 0:
            output.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end-start)
    
    return times
        
def main():
    # 1. 解析命令行参数
    args = parse_args()
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 3. 初始化模型
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
    # 4. 生成随机数据
    input_data = generate_randtext(
        batch_size = args.batch_size,
        context_length = args.context_length,
        vocab_size = args.vocab_size
    )
    input_data = input_data.to(device)
    # 5. 运行benchmark
    times = benchmark(
        model=model,
        input_data=input_data,
        warmup_steps=args.warmup_steps,
        test_steps=args.test_steps,
        mode=args.mode
    )
    # 6. 计算并打印结果（平均值和标准差）
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"平均时间: {mean_time:.4f}s, 标准差: {std_time:.4f}s")

if __name__ == "__main__":
    main()