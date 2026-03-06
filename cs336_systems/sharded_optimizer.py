import torch
import torch.distributed as dist
from torch.optim import Optimizer

class ShardedOptimizer(Optimizer):

    def __init__(self, params, optimizer_cls, **kwargs):
        """
            params: parms to optimize
            optimizer_cls: classof the optimizer(AdamW)
            **kwargs: hyperparameter for optimizer(lr, weight_decay)
        """
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.local_params = []
        self.all_params = []
        self.param_to_rank = {}
        super().__init__(params, kwargs)

        # optimizer of current rank
        self.local_optimizer = optimizer_cls(self.local_params, **kwargs)
       
        
    def add_param_group(self, param_group):
        """
            Allocate parameters according to rank and world_size
        """
        rank = dist.get_rank() 
        world_size = dist.get_world_size()
        for i, param in enumerate(param_group["params"]):
            if i % world_size == rank:
                self.local_params.append(param)
            self.all_params.append(param)
            self.param_to_rank[id(param)] = i % world_size
        super().add_param_group(param_group)
            
    def step(self, closure=None):
        loss = None if closure is None else closure()
        # update local parameters
        self.local_optimizer.step()
        # broadcast param to every rank
        for param in self.all_params:
            rank_index = self.param_to_rank[id(param)]
            dist.broadcast(param.data, src=rank_index)
        return loss