import torch
from core.base_runner import BaseRunner
from core.registry import RUNNER_REGISTRY
from core.trainer import Trainer

@RUNNER_REGISTRY.register("supervised")
class SupervisedRunner(BaseRunner):
    """Standard Supervised Learning Runner."""
    def run(self):
        self.logger.info(f"Starting Supervised Training with {self.config['epochs']} epochs...")
        
        # In a real scenario, data_loaders would be provided.
        # Here we assume self.data_loader and self.val_loader (if any) are set.
        if self.data_loader is None:
            self.logger.warning("No data_loader provided to SupervisedRunner.")
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 0.001))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        trainer = Trainer(
            model=self.model,
            train_loader=self.data_loader,
            val_loader=self.data_loader, # Using same for dummy
            optimizer=optimizer,
            device=device,
            config=self.config
        )
        
        trainer.fit()
        self.close()

    def evaluate(self):
        self.logger.info("Evaluating Supervised Model...")
        pass
