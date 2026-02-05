import os
from torch.utils.tensorboard import SummaryWriter
import logging

class ExperimentLogger:
    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(config.get("output_dir", "outputs"), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = logging.getLogger("ResearchFramework")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics, step, mode="Train"):
        log_str = f"[{mode}] Step {step} | "
        for k, v in metrics.items():
            self.writer.add_scalar(f"{mode}/{k}", v, step)
            log_str += f"{k}: {v:.4f}  "
        self.logger.info(log_str)

    def log_model_info(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param, step)
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, step)

    def close(self):
        self.writer.close()

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("ResearchFramework")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(os.path.join(output_dir, "experiment.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Stream handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger
