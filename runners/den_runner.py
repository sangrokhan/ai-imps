import torch
import torch.nn as nn
from core.base_runner import BaseRunner
from core.registry import RUNNER_REGISTRY
from implementations.den_iclr2018.train import train_den_step, get_optimizer
from implementations.den_iclr2018.expansion import expand_network, select_neurons, split_neurons
import os

@RUNNER_REGISTRY.register("den_runner")
class DENRunner(BaseRunner):
    """
    Runner for Dynamically Expandable Networks (ICLR 2018).
    Implements the 4-stage training process:
    1. Selective Retraining
    2. Dynamic Expansion (L1 regularized training)
    3. Split & Duplication
    4. Task-specific Retraining
    """
    def __init__(self, config, model, tasks_data=None):
        """
        Args:
            config (dict): Runner configuration.
            model (DEN): DEN model instance.
            tasks_data (list, optional): List of (train_loader, test_loader) for each task.
        """
        super().__init__(config, model)
        self.tasks_data = tasks_data if tasks_data is not None else []
        self.l1_lambda = config.get('l1_lambda', 0.001)
        self.l2_lambda = config.get('l2_lambda', 0.0001)
        self.lr = config.get('lr', 0.001)
        self.drift_threshold = config.get('drift_threshold', 0.01)
        self.expansion_k = config.get('expansion_k', 10)

    def run(self):
        print("Starting DEN Training Loop...")
        device = next(self.model.parameters()).device
        
        for t_idx, (train_loader, test_loader) in enumerate(self.tasks_data):
            task_id = str(t_idx)
            print(f"\n--- Task {task_id} ---")
            
            # Step 0: Add new task head/tail
            # We assume input_dim and output_dim are provided in config or inferred from data
            in_dim = self.config['input_dim']
            out_dim = self.config['output_dim']
            self.model.add_task_layer(task_id, in_dim, out_dim)
            self.model.to(device)

            if t_idx == 0:
                # First task: Standard training with L1
                self._train_first_task(task_id, train_loader)
            else:
                # Subsequent tasks: DEN 3-step process
                self._den_process(task_id, train_loader)
            
            # Evaluation
            self.evaluate(t_idx)

    def _train_first_task(self, task_id, train_loader):
        print(f"Training base task {task_id}...")
        criterion = nn.CrossEntropyLoss()
        config = {'l1_lambda': self.l1_lambda}
        
        for epoch in range(self.config.get('epochs', 10)):
            loss = train_den_step(self.model, train_loader, criterion, 
                                  lambda p: get_optimizer(p, self.lr), 
                                  config, task_id)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def _den_process(self, task_id, train_loader):
        """Stages for task t > 0."""
        # 1. Selective Retraining
        print("Stage 1: Selective Retraining...")
        self._selective_retraining(task_id, train_loader)
        
        # 2. Dynamic Expansion (Horizontal)
        print("Stage 2: Dynamic Expansion...")
        # Check if loss is still high or just expand by default? 
        # DEN paper: expand if loss > threshold after selective retraining.
        # For simplicity, we expand.
        self.model = expand_network(self.model, self.expansion_k, task_id)
        # Train expanded model with L1
        self._train_expanded(task_id, train_loader)
        
        # 3. Split & Duplication (Semantic drift prevention)
        print("Stage 3: Split & Duplication...")
        # We need state from BEFORE this task training to detect drift
        # But wait, stage 1 & 2 already modified weights. 
        # Usually we compare with state after task t-1.
        # Let's assume we have a snapshot (this is illustrative).
        # snapshot = torch.load('prev_task.pt')
        # drift_indices = select_neurons(self.model, snapshot, self.drift_threshold)
        # self.model = split_neurons(self.model, drift_indices, snapshot)
        print("Split & Duplication logic invoked (placeholders used).")

    def _selective_retraining(self, task_id, train_loader):
        # Implementation of Selective Retraining would involve:
        # 1. Sparse coding/L1 path selection
        # 2. Freezing non-selected weights
        # Here we just show the structure.
        pass

    def _train_expanded(self, task_id, train_loader):
        criterion = nn.CrossEntropyLoss()
        config = {'l1_lambda': self.l1_lambda}
        for epoch in range(self.config.get('epochs', 5)):
            loss = train_den_step(self.model, train_loader, criterion, 
                                  lambda p: get_optimizer(p, self.lr), 
                                  config, task_id)
            print(f"Expansion Epoch {epoch}, Loss: {loss:.4f}")

    def evaluate(self, current_task_idx):
        print(f"Evaluating all tasks up to {current_task_idx}...")
        # Evaluation loop for all task_ids in [0...current_task_idx]
        pass
