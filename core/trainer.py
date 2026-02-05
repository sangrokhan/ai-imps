import torch
import os
from tqdm import tqdm

class Trainer:
    """
    모델 학습 및 검증을 담당하는 범용 트레이너 클래스입니다.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.current_epoch = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch in pbar:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.model.compute_loss(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch in pbar:
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
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 모델 체크포인트 저장 로직
            if (epoch + 1) % self.config.get("save_freq", 5) == 0:
                save_path = os.path.join(self.config.get("output_dir", "outputs"), f"model_epoch_{epoch}.pth")
                self.model.save(save_path)
