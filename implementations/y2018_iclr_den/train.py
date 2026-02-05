import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(params, lr=1e-3):
    return optim.Adam(params, lr=lr)

def train_den_step(model, train_loader, criterion, optimizer_fn, config, task_id, grad_mask=None):
    """
    Performs one epoch of training for DEN with optional L1 regularization and gradient masking.
    
    Args:
        model (DEN): DEN model instance.
        train_loader (DataLoader): DataLoader for the current task.
        criterion (loss function): Loss function.
        optimizer_fn (func): Function that returns an optimizer given parameters.
        config (dict): Dictionary containing 'l1_lambda' and other hyperparameters.
        task_id (int/str): Current task ID.
        grad_mask (dict, optional): Dictionary mapping parameter names to boolean masks (True=train, False=freeze).
                                    Used for Selective Retraining.
    
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    
    # Filter parameters to optimize (only those requiring grad)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_fn(params)
    
    total_loss = 0
    device = next(model.parameters()).device
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
            
        optimizer.zero_grad()
        output = model(data, task_id)
        
        # Standard Loss
        loss = criterion(output, target)
        
        # L1 Regularization (Lasso) on weights to encourage sparsity
        l1_lambda = config.get('l1_lambda', 0.0)
        if l1_lambda > 0:
            l1_reg = torch.tensor(0., device=device)
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    l1_reg += torch.norm(param, 1)
            loss += l1_lambda * l1_reg
        
        loss.backward()
        
        # Zero-Grad Masking for Selective Retraining
        if grad_mask is not None:
            for name, param in model.named_parameters():
                if name in grad_mask:
                    # Apply mask: Gradient is multiplied by mask (0 or 1)
                    if param.grad is not None:
                        param.grad *= grad_mask[name]
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)
