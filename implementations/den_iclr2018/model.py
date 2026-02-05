import torch
import torch.nn as nn
import os
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("den")
class DEN(BaseModel):
    def __init__(self, config):
        """
        Initialize the Dynamically Expandable Network.
        
        Args:
            config (dict): Configuration dictionary containing 'hidden_dims'.
        """
        super().__init__(config)
        self.hidden_dims = config.get('hidden_dims', [64, 32])
        self.shared_layers = nn.ModuleList()
        
        if len(self.hidden_dims) > 1:
            for i in range(len(self.hidden_dims) - 1):
                self.shared_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
        
        self.task_inputs = nn.ModuleDict()  # Task-specific input heads
        self.task_outputs = nn.ModuleDict() # Task-specific output tails
        self.activation = nn.ReLU()

    def add_task_layer(self, task_id, in_dim, out_dim):
        task_id = str(task_id)
        if task_id not in self.task_inputs:
            self.task_inputs[task_id] = nn.Linear(in_dim, self.hidden_dims[0])
            self.task_outputs[task_id] = nn.Linear(self.hidden_dims[-1], out_dim)

    def forward(self, x, **kwargs):
        task_id = str(kwargs.get('task_id', 0))
        if task_id not in self.task_inputs:
            # Fallback to first task if not found, or raise error
            task_id = list(self.task_inputs.keys())[0] if self.task_inputs else None
            if task_id is None:
                raise KeyError("No tasks initialized in DEN.")
        
        out = self.task_inputs[task_id](x)
        out = self.activation(out)
        
        for layer in self.shared_layers:
            out = layer(out)
            out = self.activation(out)
            
        out = self.task_outputs[task_id](out)
        return out
