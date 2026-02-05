from abc import ABC, abstractmethod

class BaseRunner(ABC):
    """Abstract Base Class for training or interaction loops."""
    def __init__(self, config, model, data_loader=None, env=None):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.env = env

    @abstractmethod
    def run(self):
        """Main execution logic."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation logic."""
        pass
