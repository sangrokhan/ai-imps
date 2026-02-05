import torch
import torch.nn as nn
import os
import copy

class DEN(nn.Module):
    def __init__(self, hidden_dims):
        """
        Initialize the Dynamically Expandable Network.
        
        Args:
            hidden_dims (list): List of integers representing the number of neurons in each shared hidden layer.
                                For example, [64, 32] creates a shared structure of 64 -> 32.
                                Input will connect to 64, Output will connect from 32.
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.shared_layers = nn.ModuleList()
        
        # Create shared hidden layers
        # If hidden_dims = [h1, h2, h3], layers are h1->h2, h2->h3
        if len(hidden_dims) > 1:
            for i in range(len(hidden_dims) - 1):
                self.shared_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        self.task_inputs = nn.ModuleDict()  # Task-specific input heads
        self.task_outputs = nn.ModuleDict() # Task-specific output tails
        self.activation = nn.ReLU() # Using ReLU by default

    def add_task_layer(self, task_id, in_dim, out_dim):
        """
        Dynamically adds specific input and output layers for a new task.
        """
        task_id = str(task_id)
        if task_id not in self.task_inputs:
            # Input layer: in_dim -> first hidden layer
            self.task_inputs[task_id] = nn.Linear(in_dim, self.hidden_dims[0])
            # Output layer: last hidden layer -> out_dim
            self.task_outputs[task_id] = nn.Linear(self.hidden_dims[-1], out_dim)

    def forward(self, x, task_id):
        """
        Forward pass for a specific task.
        """
        task_id = str(task_id)
        if task_id not in self.task_inputs:
            raise KeyError(f"Task {task_id} not initialized. Call add_task_layer first.")
        
        # 1. Task-specific Input
        out = self.task_inputs[task_id](x)
        out = self.activation(out)
        
        # 2. Shared Layers
        for layer in self.shared_layers:
            out = layer(out)
            out = self.activation(out)
            
        # 3. Task-specific Output
        out = self.task_outputs[task_id](out)
        return out

    def save_weights(self, task_id, save_dir):
        """
        Saves the current state of the model for a specific task.
        """
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{task_id}.pt")
        torch.save({
            'state_dict': self.state_dict(),
            'hidden_dims': self.hidden_dims
        }, path)
        print(f"Weights saved to {path}")

    def load_weights(self, task_id, save_dir):
        """
        Loads weights for a specific task. 
        Note: This handles simple loading. If architecture changed (expansion), 
        logic needs to be robust (often strict=False is a starting point).
        """
        path = os.path.join(save_dir, f"{task_id}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path)
            # In a full implementation, we might need to adjust self.shared_layers 
            # if checkpoint['hidden_dims'] differs from current self.hidden_dims
            # For now, we assume we want to load what we can.
            
            # TODO: Handle architecture mismatch if network was expanded in saved state
            if checkpoint['hidden_dims'] != self.hidden_dims:
                print(f"Warning: Loaded hidden dims {checkpoint['hidden_dims']} differ from current {self.hidden_dims}")
                # Ideally, we would reconstruct layers here to match loaded state
            
            self.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Weights loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")
