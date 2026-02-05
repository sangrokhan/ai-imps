
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import ResNet18APD
from apd_layers import APDLinear, APDConv2d

def get_split_cifar100(task_id, batch_size=64, root='./data'):
    """
    Returns data loaders for a specific task of Split-CIFAR100.
    Each task contains 10 classes.
    Task 0: 0-9, Task 1: 10-19, ...
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    start_class = task_id * 10
    end_class = (task_id + 1) * 10
    
    train_indices = [i for i, label in enumerate(trainset.targets) if start_class <= label < end_class]
    test_indices = [i for i, label in enumerate(testset.targets) if start_class <= label < end_class]
    
    train_loader = DataLoader(Subset(trainset, train_indices), batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(Subset(testset, test_indices), batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def freeze_previous_tasks(model, current_task_id):
    """
    Freezes weight_task/masks corresponding to previous tasks.
    Keeps shared weights and current task weights trainable.
    """
    for name, param in model.named_parameters():
        # param name example: layer1.0.conv1.weight_task.0
        # or layer1.0.conv1.task_masks_task.0
        
        # By default enable gradient
        if 'weight_shared' in name or 'bias' in name:
            param.requires_grad = True # Always trainable (but regularized)
        elif 'weight_task' in name or 'task_masks' in name:
            # Check which task ID this parameter belongs to
            # The structure is usually module.weight_task.<task_id_str>
            parts = name.split('.')
            # Assuming the last part is the task id key
            tid = parts[-1]
            if tid == str(current_task_id):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 100 classes total for CIFAR-100
    model = ResNet18APD(num_classes=100).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # History of accuracies
    acc_matrix = np.zeros((args.tasks, args.tasks))

    for task_id in range(args.tasks):
        print(f"\n=== Starting Task {task_id} ===")
        model.add_task(task_id)
        model.to(device) # ensure new params are on device
        
        train_loader, test_loader = get_split_cifar100(task_id, batch_size=args.batch_size)
        
        freeze_previous_tasks(model, task_id)
        
        # Filter trainable parameters
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=args.lr, momentum=0.9, weight_decay=1e-4) # Regular weight decay for general regularization
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs, task_id)
                
                # Standard Classification Loss
                loss_cls = criterion(outputs, targets)
                
                # Sparsity penalty on Masks (L1)
                loss_sparsity = 0
                count_masks = 0
                for name, param in model.named_parameters():
                    if 'task_masks' in name and str(task_id) in name:
                         loss_sparsity += torch.sum(torch.abs(param))
                         count_masks += 1
                
                loss = loss_cls + args.lambda_sparse * loss_sparsity
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if args.debug and batch_idx >= 5:
                    break
            
            print(f"Task {task_id} | Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

        # Evaluation on all tasks seen so far
        print(f"--- Evaluation after Task {task_id} ---")
        model.eval()
        with torch.no_grad():
            for t in range(task_id + 1):
                _, t_test_loader = get_split_cifar100(t, batch_size=args.batch_size)
                correct = 0
                total = 0
                for inputs, targets in t_test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs, t)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total
                acc_matrix[task_id, t] = acc
                print(f"Task {t} Accuracy: {acc:.2f}%")

    print("\n=== Final Results ===")
    print("Accuracy Matrix:")
    print(acc_matrix)
    
    # Calculate Average Accuracy and Forgetting
    avg_acc = np.mean(acc_matrix[args.tasks-1, :args.tasks])
    forgetting = 0
    if args.tasks > 1:
        for i in range(args.tasks - 1):
            max_acc_prev = np.max(acc_matrix[:args.tasks-1, i])
            current_acc = acc_matrix[args.tasks-1, i]
            forgetting += (max_acc_prev - current_acc)
        forgetting /= (args.tasks - 1)
        
    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"Average Forgetting: {forgetting:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per task')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tasks', type=int, default=10, help='Number of tasks to run')
    parser.add_argument('--lambda_sparse', type=float, default=1e-5, help='Sparsity penalty weight')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (fewer batches)')
    
    args = parser.parse_args()
    train(args)
