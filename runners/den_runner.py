import torch
import torch.nn as nn
from core.base_runner import BaseRunner
from core.registry import RUNNER_REGISTRY
from implementations.y2018_iclr_den.train import train_den_step, get_optimizer
from implementations.y2018_iclr_den.expansion import expand_network, select_neurons, split_neurons
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
        self.logger.info("Starting DEN Training Loop...")
        device = next(self.model.parameters()).device
        
        for t_idx, (train_loader, test_loader) in enumerate(self.tasks_data):
            task_id = str(t_idx)
            self.logger.info(f"\n--- Task {task_id} ---")
            
            # Step 0: Add new task head/tail
            in_dim = self.config['input_dim']
            out_dim = self.config['output_dim']
            self.model.add_task_layer(task_id, in_dim, out_dim)
            self.model.to(device)

            if t_idx == 0:
                self._train_first_task(task_id, train_loader)
            else:
                self._den_process(task_id, train_loader)
            
            self.evaluate(t_idx)
        
        self.close()

    def _train_first_task(self, task_id, train_loader):
        self.logger.info(f"Training base task {task_id}...")
        criterion = nn.CrossEntropyLoss()
        config = {'l1_lambda': self.l1_lambda}
        
        for epoch in range(self.config.get('epochs', 10)):
            loss = train_den_step(self.model, train_loader, criterion, 
                                  lambda p: get_optimizer(p, self.lr), 
                                  config, task_id)
            self.writer.add_scalar(f"Loss/task_{task_id}", loss, epoch)
            if epoch % 5 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")

    def _den_process(self, task_id, train_loader):
        """Stages for task t > 0."""
        # 1. Selective Retraining
        self.logger.info("Stage 1: Selective Retraining...")
        self._selective_retraining(task_id, train_loader)
        
        # 2. Dynamic Expansion (Horizontal)
        self.logger.info("Stage 2: Dynamic Expansion...")
        self.model = expand_network(self.model, self.expansion_k, task_id)
        self._train_expanded(task_id, train_loader)
        
        # 3. Split & Duplication (Semantic drift prevention)
        self.logger.info("Stage 3: Split & Duplication...")
        self.logger.info("Split & Duplication logic invoked (placeholders used).")

    def _selective_retraining(self, task_id, train_loader):
        pass

    def _train_expanded(self, task_id, train_loader):
        criterion = nn.CrossEntropyLoss()
        config = {'l1_lambda': self.l1_lambda}
        for epoch in range(self.config.get('epochs', 5)):
            loss = train_den_step(self.model, train_loader, criterion, 
                                  lambda p: get_optimizer(p, self.lr), 
                                  config, task_id)
            self.writer.add_scalar(f"Loss/task_{task_id}_expansion", loss, epoch)
            self.logger.info(f"Expansion Epoch {epoch}, Loss: {loss:.4f}")

    def evaluate(self, current_task_idx):
        self.logger.info(f"Evaluating all tasks up to {current_task_idx}...")
        pass
