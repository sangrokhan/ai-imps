import torch
import os
from core.logger import ExperimentLogger

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.current_epoch = 0
        self.logger = ExperimentLogger(config)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for i, batch in enumerate(self.train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.model.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if i % self.config.get("log_interval", 10) == 0:
                step = self.current_epoch * len(self.train_loader) + i
                self.logger.log_scalar("Train/BatchLoss", loss.item(), step)
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        for batch in self.val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.model.compute_loss(outputs, targets)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self):
        epochs = self.config.get("epochs", 10)
        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.logger.log_metrics({"Loss": train_loss, "ValLoss": val_loss}, epoch, mode="Epoch")
            self.logger.log_model_info(self.model, epoch)
            if (epoch + 1) % self.config.get("save_freq", 5) == 0:
                save_path = os.path.join(self.config.get("output_dir", "outputs"), f"model_epoch_{epoch}.pth")
                self.model.save(save_path)
        self.logger.close()
