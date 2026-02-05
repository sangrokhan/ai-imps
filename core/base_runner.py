from abc import ABC, abstractmethod
import os
from core.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter

class BaseRunner(ABC):
    """Abstract Base Class for training or interaction loops."""
    def __init__(self, config, model, data_loader=None, env=None):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.env = env
        
        # Setup Logging
        self.output_dir = config.get("output_dir", "outputs")
        self.logger = setup_logger(self.output_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))
        
        self.logger.info(f"Runner initialized. Logs and TensorBoard data will be in {self.output_dir}")

    @abstractmethod
    def run(self):
        """Main execution logic."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation logic."""
        pass
        
    def close(self):
        """Cleanup."""
        self.writer.close()
        self.logger.info("Runner session closed.")
