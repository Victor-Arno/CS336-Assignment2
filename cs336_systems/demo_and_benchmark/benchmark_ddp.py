import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
from cs336_systems.ddp import (
      DDPIndividualParameters, DDPAsyncIndividualParameters,
      on_after_backward, on_after_backward_flat, on_after_backward_async
  )
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def worker(rank, ddp_class, fn, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # base parameters
    batch_size = 1
    context_length = 256
    vocab_size = 256
    input_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    input_labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    warm_up = 5
    test_step = 10
    total_list = []
    comm_list = []
    
    # model and optimizer
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        rope_theta=10000.0,
    ).to(device)
    
    # xl model
#    model = BasicsTransformerLM(
#        vocab_size=50257,
#        context_length=256,
#        d_model=1600,
#        d_ff=6400,
#        num_layers=48,
#        num_heads=25,
#        rope_theta=10000.0,
#    ).to(device)
    
    ddp_model =  ddp_class(model)
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = AdamW(
        params=ddp_model.module.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    
    for _ in range(warm_up):
        optimizer.zero_grad()
        output_data = ddp_model(input_data)
        loss = loss_fn(output_data.view(-1, output_data.size(-1)), input_labels.view(-1))
        loss.backward()
        fn(ddp_model)
        optimizer.step()
    
            
    torch.cuda.synchronize()
            
    for _ in range(test_step):
        # total start time
        torch.cuda.synchronize()
        total_start = time.perf_counter()
        
        optimizer.zero_grad()
        output = ddp_model(input_data)

        loss = loss_fn(output.view(-1, output.size(-1)), input_labels.view(-1))
        loss.backward()
        # communicate start time
        torch.cuda.synchronize()
        comm_start = time.perf_counter()
        
        fn(ddp_model)
        
        # communicate end time
        torch.cuda.synchronize()
        comm_end = time.perf_counter()
        
        optimizer.step()
        torch.cuda.synchronize()
        # total start time
        total_end = time.perf_counter()
        
        comm_list.append(comm_end-comm_start)
        total_list.append(total_end-total_start)
    
    if rank == 0:
        print(f"Class: {ddp_class.__name__}")
        print(f"Method: {fn.__name__}")
        avg_total = sum(total_list) / len(total_list)
        avg_comm = sum(comm_list) / len(comm_list)
        print(f"Avg total time: {avg_total:.4f}s")
        print(f"Avg comm time: {avg_comm:.4f}s")
        print(f"comm portion: {avg_comm/avg_total*100:.1f}%")
        
    dist.destroy_process_group()
    
def main():
    world_size = 2
    configs = [
        (DDPIndividualParameters, on_after_backward),
        (DDPIndividualParameters, on_after_backward_flat),
        (DDPAsyncIndividualParameters, on_after_backward_async),
    ]
    for ddp_class, fn in configs:
        mp.spawn(worker, args=(ddp_class, fn, world_size), nprocs=world_size)
        print('-'*100)

if __name__=="__main__":
    main()
    
    