import torch
import torch.nn as nn
from model import DEN
from expansion import expand_network, select_neurons, split_neurons
from train import train_den_step, get_optimizer
import copy
import os

def test_den_flow():
    print("Initializing DEN...")
    model = DEN(hidden_dims=[10, 5])
    
    # Task 1
    print("Adding Task 1...")
    model.add_task_layer(task_id=1, in_dim=8, out_dim=2)
    
    # Dummy Data
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 0, 1])
    
    # Train Task 1
    print("Training Task 1...")
    optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.1)
    criterion = nn.CrossEntropyLoss()
    config = {'l1_lambda': 0.001}
    
    train_loader = [(x, y)]
    
    for _ in range(5):
        loss = train_den_step(model, train_loader, criterion, optimizer_fn, config, task_id=1)
        print(f"Task 1 Loss: {loss:.4f}")
        
    # Save weights
    print("Saving weights...")
    model.save_weights(task_id=1, save_dir='./checkpoints')
    
    old_state = copy.deepcopy(model.state_dict())
    
    # Task 2
    print("\nAdding Task 2...")
    model.add_task_layer(task_id=2, in_dim=8, out_dim=2)
    
    # Simulate training Phase for Task 2 (which changes shared weights)
    print("Simulating Task 2 Training (Drift)...")
    with torch.no_grad():
        # Introduce significant drift in shared_layers[0] (affects hidden[1])
        # shared_layers[0] is [10 -> 5] (Wait, shared_layers starts from hidden[0]=10 to hidden[1]=5)
        # So it is 5x10 matrix.
        # Drift in row i means neuron i in hidden[1] drifted.
        model.shared_layers[0].weight.data[0, :] += 0.5 # Drift neuron 0 in hidden layer 1 (output of shared[0])
        
    # Check Drift
    print("Checking Drift...")
    # threshold 0.1 should catch the 0.5 change
    drift_indices = select_neurons(model, old_state, threshold=0.1)
    print(f"Drift Indices: {drift_indices}")
    
    # Split
    if drift_indices:
        print("Splitting Neurons...")
        model = split_neurons(model, drift_indices, old_state)
        print(f"New Hidden Dims: {model.hidden_dims}")
        
        # Verify shapes
        # If we split 1 neuron in hidden[1] (output of shared[0]):
        # shared[0] out_features should be 5+1 = 6.
        # shared[1] (if exists) in_features should be 6.
        # task_outputs input should be 6.
        print(f"Shared Layer 0 shape: {model.shared_layers[0].weight.shape}")
        if len(model.shared_layers) > 1:
             print(f"Shared Layer 1 shape: {model.shared_layers[1].weight.shape}")
        else:
             print(f"Task Output shape: {model.task_outputs['2'].weight.shape}")
        
    # Expand
    print("\nExpanding Network...")
    model = expand_network(model, k=2, task_id=2)
    print(f"New Hidden Dims after expansion: {model.hidden_dims}")
    
    # Verify forward pass
    print("Verifying Forward Pass Task 2...")
    out = model(x, task_id=2)
    print("Output shape:", out.shape)
    assert out.shape == (4, 2)
    print("Test passed!")

if __name__ == "__main__":
    test_den_flow()
