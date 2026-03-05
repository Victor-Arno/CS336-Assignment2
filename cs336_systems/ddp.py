import torch
import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDPIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        # save original model
        self.module = module
        # broadcast parameters of rank 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
                
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DDPBucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.handles = []

        # broadcast parameters of rank 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # divide parameters into buckets and register hooks in one pass
        self.buckets = [[]]
        self.param_to_bucket = {}
        current_size = 0
        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
            param_size_mb = (param.numel() * param.element_size()) / (1024 * 1024)
            if param_size_mb + current_size > bucket_size_mb and len(self.buckets[-1]) > 0:
                self.buckets.append([])
                current_size = 0
            self.buckets[-1].append(param)
            self.param_to_bucket[id(param)] = len(self.buckets) - 1
            current_size += param_size_mb
            param.register_post_accumulate_grad_hook(self._hook_fn)

        self.bucket_ready_count = [0] * len(self.buckets)

    def _hook_fn(self, param):
        bucket_idx = self.param_to_bucket[id(param)]
        self.bucket_ready_count[bucket_idx] += 1
        if self.bucket_ready_count[bucket_idx] == len(self.buckets[bucket_idx]):
            grads = [p.grad.data for p in self.buckets[bucket_idx]]
            flat = _flatten_dense_tensors(grads)
            handle = dist.all_reduce(flat, async_op=True)
            self.handles.append((handle, flat, bucket_idx))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, flat, bucket_idx in self.handles:
            handle.wait()
            flat /= dist.get_world_size()
            grads = [p.grad.data for p in self.buckets[bucket_idx]]
            for param, new_grad in zip(self.buckets[bucket_idx], _unflatten_dense_tensors(flat, grads)):
                param.grad.data = new_grad

    def reset(self):
        self.handles.clear()
        self.bucket_ready_count = [0] * len(self.buckets)       
    
class DDPAsyncIndividualParameters(nn.Module):
      def __init__(self, module):
          super().__init__()
          # save original model
          self.module = module
          # save handle list
          self.handles = []
          # broadcast parameters of rank 0
          for param in self.module.parameters():
              dist.broadcast(param.data, src=0)
              # hook register
              if param.requires_grad:
                  param.register_post_accumulate_grad_hook(
                      lambda p: self.handles.append(
                          dist.all_reduce(p.grad.data,async_op=True)
                      )
                  )
                  
      def forward(self, *args, **kwargs):
          return self.module(*args, **kwargs)

      def finish_gradient_synchronization(self):
          for handle in self.handles:
              handle.wait()
         
          for param in self.module.parameters():
              if param.grad is not None:
                   param.grad.data /= dist.get_world_size() 
        
# async communication
def on_after_backward_async(model):
      model.finish_gradient_synchronization()
      
# naive ddp
def on_after_backward(model):
    for param in model.module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data)
            param.grad.data /= dist.get_world_size() 
            
# minimal ddp
def on_after_backward_flat(model):
    grad_list = []
    params_with_grad = []
    for param in model.module.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.data)
            params_with_grad.append(param)
            
    Flatten_Tensor = _flatten_dense_tensors(grad_list)
    dist.all_reduce(Flatten_Tensor)
    Flatten_Tensor /= dist.get_world_size()
    for param, new_grad in zip(params_with_grad, _unflatten_dense_tensors(Flatten_Tensor, grad_list)):
        if param.grad is not None:
            param.grad.data = new_grad