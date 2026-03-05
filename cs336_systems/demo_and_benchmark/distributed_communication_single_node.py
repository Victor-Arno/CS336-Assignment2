import os
import math
import time
import torch
import torch.multiprocessing as mp
import pandas as pd
import torch.distributed as dist

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        

def worker(rank, world_size, backend, num_elements, result_queue):
    setup(rank, world_size, backend)
    
    device = "cpu" if backend == "gloo" else "cuda"
    data = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    # warmp up
    for _ in range(5):
        dist.all_reduce(data)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # benchmark
    num_iters = 10
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        # calculate time
        dist.all_reduce(data)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_iters
    
    # collect timing from all ranks
    all_times = [0.0] * world_size
    dist.all_gather_object(all_times, elapsed)
    
    # rank 
    if rank == 0:
        avg_time = sum(all_times) / world_size
        result_queue.put(avg_time)
        
    
    dist.destroy_process_group()

def main():
    backends = [("gloo", "cpu"), ("nccl", "cuda")]
    size_mbs = [1, 10, 100, 1000]
    num_procs_list = [1, 2, 4]
    results = []
    for backend, device in backends:
        for num_procs in num_procs_list:
            for size_mb in size_mbs:
                num_elements = size_mb * 1024 * 1024 // 4
                result_queue = mp.get_context("spawn").Queue()
                mp.spawn(worker, args=(num_procs, backend, num_elements, result_queue),nprocs=num_procs)
                avg_time = result_queue.get()
                # get results from queue
                results.append({
                    'backend': backend,
                    'num_procs': num_procs,
                    'size_mb': size_mb,
                    'time(s)': f'{avg_time:.4f}',
                })
                print(f"done: backend={backend} procs={num_procs} size={size_mb}MB time={avg_time:.4f}s")
                
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
if __name__ == "__main__":
    main()