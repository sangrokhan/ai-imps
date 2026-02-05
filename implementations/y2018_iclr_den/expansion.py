import torch
import torch.nn as nn
from copy import deepcopy

def expand_network(model, k, task_id):
    """
    Expands the network by adding 'k' units to each hidden layer.
    
    Args:
        model (DEN): The DEN model to expand.
        k (int): Number of neurons to add to each hidden layer.
        task_id (int/str): The current task ID (to expand specific heads/tails if needed, 
                           though only shared layers are typically expanded horizontally).
                           
    Returns:
        DEN: The expanded model (in-place modification mostly, but returning for safety).
    """
    if k <= 0:
        return model
        
    device = next(model.parameters()).device
    
    # Iterate over shared layers
    new_hidden_dims = []
    
    # We need to handle the connections:
    # Layer i: Input -> Output
    # If we expand Layer i output by k, we must expand Layer i+1 input by k.
    
    # 1. Expand Shared Layers
    for i, layer in enumerate(model.shared_layers):
        old_in_dim = layer.in_features
        old_out_dim = layer.out_features
        
        # New dimensions
        # If it's the first shared layer, input depends on previous task_input output (which is hidden_dims[0])
        # Wait, model.task_inputs[task_id] output dim IS hidden_dims[0].
        # So if we expand hidden_dims[0] by k, we must expand task_inputs[task_id] output by k. (Actually ALL task inputs?)
        # DEN paper: "capacity expansion". Usually we add units that are *only* used by the new task initially or sparse?
        # For simplicity, we assume we expand the shared capacity for *all*. but we might only connect the new task to them.
        # But `expand_network` changes dimensions, so all matrices must reshape.
        
        new_out_dim = old_out_dim + k
        
        # However, for the input dim of layer i:
        # If i == 0: input is from task_input. task_input output must match layer 0 input.
        # If we expand layer 0 input, we basically say hidden_dims[0] increased.
        # So layer 0 takes input from 'prev layer'.
        
        # Let's handle it layer by layer.
        # We assume we want to expand ALL hidden dimensions by k.
        pass

    # Implementation Strategy: Rebuild layers one by one.
    
    # Update hidden_dims first
    old_hidden_dims = list(model.hidden_dims)
    new_hidden_dims = [d + k for d in old_hidden_dims]
    model.hidden_dims = new_hidden_dims
    
    # 1. Update Task Inputs (heads) -> Connect to expanded first hidden layer
    # The output of task_inputs must match new_hidden_dims[0]
    for tid, head in model.task_inputs.items():
        old_head_weight = head.weight.data # [old_h0, in_dim]
        old_head_bias = head.bias.data     # [old_h0]
        
        in_dim = head.in_features
        new_head = nn.Linear(in_dim, new_hidden_dims[0])
        
        # Copy old weights
        new_head.weight.data[:old_hidden_dims[0], :] = old_head_weight
        new_head.bias.data[:old_hidden_dims[0]] = old_head_bias
        
        # Initialize new weights (e.g. zero or random) to break symmetry? 
        # Usually zero for "new" units to start inactive, or small random.
        # Zero initialization is safer to not disturb existing behavior if we rely on sparsity.
        # But for training, random is better.
        # Let's init new rows with xavier/kaiming
        nn.init.xavier_uniform_(new_head.weight.data[old_hidden_dims[0]:, :])
        new_head.bias.data[old_hidden_dims[0]:] = 0
        
        model.task_inputs[tid] = new_head.to(device)

    # 2. Update Shared Layers
    new_shared_layers = nn.ModuleList()
    
    # To handle connections between layers:
    # Layer 0: old_h0 -> old_h1  ===> new_h0 -> new_h1
    
    for i, layer in enumerate(model.shared_layers):
        # old dims
        d_in_old = layer.in_features
        d_out_old = layer.out_features
        
        d_in_new = new_hidden_dims[i]      # Expanded input from prev layer
        d_out_new = new_hidden_dims[i+1] if i < len(model.shared_layers) - 1 else new_hidden_dims[-1]
        
        # Wait, logic check:
        # model.shared_layers has len(hidden_dims)-1 layers?
        # If hidden_dims = [64, 32]. shared_layers has 1 layer (64->32).
        # i=0. d_in=64, d_out=32.
        # new: 74->42 (k=10).
        
        # If hidden_dims = [64]. shared_layers is empty.
        
        new_layer = nn.Linear(d_in_new, d_out_new)
        
        old_weight = layer.weight.data # [d_out_old, d_in_old]
        old_bias = layer.bias.data     # [d_out_old]
        
        # Copy quadrant: old -> old
        new_layer.weight.data[:d_out_old, :d_in_old] = old_weight
        new_layer.bias.data[:d_out_old] = old_bias
        
        # Initialize other quadrants
        # New Output, Old Input (bottom-left)
        nn.init.xavier_uniform_(new_layer.weight.data[d_out_old:, :d_in_old]) 
        # Old Output, New Input (top-right)
        nn.init.xavier_uniform_(new_layer.weight.data[:d_out_old, d_in_old:])
        # New Output, New Input (bottom-right)
        nn.init.xavier_uniform_(new_layer.weight.data[d_out_old:, d_in_old:])
        
        # Init new biases
        new_layer.bias.data[d_out_old:] = 0
        
        new_shared_layers.append(new_layer.to(device))
        
    model.shared_layers = new_shared_layers

    # 3. Update Task Outputs (tails) -> Connect from expanded last hidden layer
    # Input of task_outputs must match new_hidden_dims[-1]
    for tid, tail in model.task_outputs.items():
        old_tail_weight = tail.weight.data # [out_dim, old_last_h]
        # bias is [out_dim], no change in size
        
        out_dim = tail.out_features
        new_tail = nn.Linear(new_hidden_dims[-1], out_dim)
        
        # Copy old weights (columns match old hidden)
        new_tail.weight.data[:, :old_hidden_dims[-1]] = old_tail_weight
        new_tail.bias.data[:] = tail.bias.data
        
        # Init new weights (columns from new hidden units)
        nn.init.xavier_uniform_(new_tail.weight.data[:, old_hidden_dims[-1]:])
        
        model.task_outputs[tid] = new_tail.to(device)
        
    print(f"Network expanded: {old_hidden_dims} -> {new_hidden_dims}")
    return model

def select_neurons(model, old_state_dict, threshold=0.01):
    """
    Identifies neurons that have drifted beyond a threshold.
    
    Args:
        model (DEN): Current model with updated weights.
        old_state_dict (dict): Snapshot of model weights before training the current task.
        threshold (float): Drift threshold (epsilon).
        
    Returns:
        dict: A dictionary mapping layer_index to a list of neuron indices to split.
    """
    drift_indices = {}
    
    for i, layer in enumerate(model.shared_layers):
        # Layer weight name in state_dict (assuming Sequential or ModuleList naming)
        # model.shared_layers is a ModuleList, so keys are "shared_layers.0.weight", etc.
        # But here we have the layer object directly.
        
        # We need to compare layer.weight with old_state_dict corresponding entry.
        weight_key = f"shared_layers.{i}.weight"
        if weight_key not in old_state_dict:
            continue
            
        old_weight = old_state_dict[weight_key] # [out_dim, in_dim]
        new_weight = layer.weight.data          # [out_dim, in_dim]
        
        # Drift is L2 distance of the weight vector for each neuron (row in weight matrix)
        # Note: If layer size changed (expanded), comparing shapes is tricky.
        # "Split & Duplication" happens usually *after* L1 training but *before* expansion? 
        # Or mixed? 
        # Usually: Train -> Select Neurons -> Split -> Retrain/Expand.
        # If we already expanded, the shapes match? 
        # Let's assume we compare the *intersection* or assumption is sizes didn't change YET,
        # or we only check the 'old' neurons.
        
        # We only check drift for the neurons that existed before (up to old_weight.shape[0]).
        
        num_existing = old_weight.shape[0]
        
        # Calculate L2 drift for each neuron row
        # diff: [num_existing, in_dim]
        # We must truncate new_weight columns too if input expanded?
        # Assuming we check drift on the 'common' connections.
        
        common_in = min(old_weight.shape[1], new_weight.shape[1])
        diff = new_weight[:num_existing, :common_in] - old_weight[:, :common_in]
        
        l2_drift = torch.norm(diff, p=2, dim=1) # [num_existing]
        
        drifting_neurons = torch.where(l2_drift > threshold)[0].tolist()
        
        if drifting_neurons:
            drift_indices[i] = drifting_neurons
            
    return drift_indices

def split_neurons(model, drift_indices, old_state_dict):
    """
    Splits the identified neurons.
    
    Args:
        model (DEN): Model to modify.
        drift_indices (dict): {layer_index: [neuron_indices]}.
        old_state_dict (dict): To restore weights for one of the copies.
    """
    if not drift_indices:
        return model
        
    device = next(model.parameters()).device
    
    # We process layer by layer. Note that splitting a neuron in layer i 
    # affects Layer i's output dim and Layer i+1's input dim.
    # Handling indices shifts if we do it sequentially.
    
    # It's easier to compute new configurations first and rebuild.
    
    new_hidden_dims = list(model.hidden_dims)
    
    # Calculate how many splits per layer
    added_counts = [0] * len(model.shared_layers)
    for i, indices in drift_indices.items():
        added_counts[i] = len(indices)
        new_hidden_dims[i] += len(indices) # Wait, hidden_dims[i] corresponds to output of shared_layers[i] IS IT?
        # hidden_dims = [h1, h2]. shared_layers: [in->h1, h1->h2]. 
        # shared_layers[0] output is h1. shared_layers[0] index is 0.
        # So yes, hidden_dims[i] is output size of shared_layers[i].
        # (Assuming shared_layers[0] connects task_input -> h1? NO.)
        # Wait, architecture:
        # task_input: in -> h0.
        # shared_layers[0]: h0 -> h1.
        # ...
        # So shared_layers is between hidden layers.
        # If len(shared_layers) == len(hidden_dims) - 1.
        # Then shared_layers[i] output is hidden_dims[i+1].
        # We need to be careful with indexing.
        pass
        
    # Re-evaluating indices:
    # hidden_dims = [h0, h1, h2].
    # task_input: -> h0.
    # shared[0]: h0 -> h1.
    # shared[1]: h1 -> h2.
    # task_output: h2 -> out.
    
    # If we split neurons in h0 (layer index -1? or input layer?):
    # Usually "Split" applies to all hidden units.
    # In my current mapping:
    # h0 is output of task_input.
    # h1 is output of shared[0].
    # h2 is output of shared[1].
    
    # drift_indices usually comes from `shared_layers`.
    # If drift in shared_layers[0] (h0 -> h1), it means neurons in h1 drifted? 
    # Yes, row `j` in W corresponds to neuron `j` in the output layer (h1).
    # So drift_indices[0] means drifts in h1.
    
    # What about h0? h0 is output of task_input. 
    # We might need to check drift in task_input? 
    # But usually task_input is task-specific, so it shouldn't drift from "old" tasks because it's new.
    # Wait, if we retrain, task_input for OLD tasks might change?
    # BUT we only train the NEW task's head. The old task heads are frozen or not used.
    # So h0 units (shared input units) might drift because of the connection from the NEW head.
    # Actually, h0 is shared. The weights feeding INTO h0 from NEW head are new.
    # But the weights feeding OUT of h0 (in shared[0]) might drift?
    # YES. shared[0] takes h0 as input. 
    # If we are talking about neurons in h0, we look at their *incoming* weights?
    # Incoming to h0 is task_input.
    # If task_input is new, there is no "old" weight to compare.
    
    # So, typically Split applies to Hidden Layers where "catastrophic forgetting" happens.
    # Forgetting happens when we change weights of Shared Layers.
    # The weights of shared_layers[0] (h0 -> h1) change.
    # This affects how h0 is used? and how h1 activates?
    # If row j of shared_layers[0] changes, it means neuron j in h1 has changed behavior.
    # So we split neuron j in h1.
    
    # What about neuron in h0?
    # Weights affecting h0 are in task_inputs.
    # Since we have task-specific heads, h0 drift isn't "shared weight drift" in the same sense?
    # The paper says: "We inspect the drift of each neuron in the network."
    # We'll focus on the outputs of shared_layers (h1, h2...). 
    # And maybe we treat the input of shared_layers[0] (h0) as fixed? 
    # Or maybe h0 is also checked?
    # If h0 changes, it means we need to split units in h0?
    # To split h0, we need to look at incoming weights to h0.
    # But incoming is task-specific.
    # Let's simplify: We split neurons in the shared hidden layers [h0, h1...].
    # My shared_layers list implements connections between them.
    # shared_layers[0] defines h1 (output).
    # What defines h0? Ideally, nothing in `shared_layers`. 
    # h0 is implicitly defined by `task_inputs` output dim and `shared_layers[0]` input dim.
    
    # If we want to split h0:
    # We need to update `task_inputs` (all of them) and `shared_layers[0]` (input dim).
    # And we check drift in...? 
    # DEN usually has a "base" network.
    # Let's assume we proceed with splitting `shared_layers` outputs (h1, h2...).
    # If we need to split h0, we might need logic for that.
    # But strictly, h0 is an "input" to the shared stack.
    
    # Let's implement splitting for outputs of shared_layers (h1..h_last).
    # And we should also consider h0 if we treat it as a hidden layer.
    # But h0 is "produced" by task_input.
    # The drift of h0 would be defined by changes in `task_input`? 
    # But `task_input` is new for new task.
    # So h0 drift is only relevant if we RETRAIN old task inputs? We don't.
    # So h0 shouldn't drift? 
    # WAIT. Optimization updates `shared_layers`.
    # `shared_layers[0]` connects h0 -> h1.
    # Updates to `shared_layers[0]` weights change how h0 is *used* to produce h1.
    # This is drift of h1 neurons (rows of shared_layers[0]).
    # So we are correct to look at `shared_layers` weights.
    
    # Implementation Detail:
    # Iterate shared layers.
    # If layer i has drift_indices:
    #   The output dim of layer i (which is hidden_dims[i+1] in my indexing if i start at 0->h1?)
    #   Wait, my indexing:
    #   hidden_dims = [h0, h1, h2]
    #   shared_layers[0]: h0 -> h1.
    #   layer.weight is [h1, h0].
    #   Drift in row j means neuron j in h1 changed.
    #   So we split neuron j in h1.
    #   This increases h1 size.
    #   This affects shared_layers[0] (output dim) and shared_layers[1] (input dim).
    
    new_shared_layers = nn.ModuleList()
    
    # We also need to update hidden_dims.
    # hidden_dims indices: 0 -> h0. 1 -> h1.
    # shared_layers[i] output is hidden_dims[i+1].
    
    current_h_dims = list(model.hidden_dims)
    
    # Map from old_layer_index -> list of split indices
    # We need to reconstruct layer by layer.
    
    # 1. Update h0? we assume h0 doesn't split for now (simplification).
    
    # reconstruction loop
    for i, layer in enumerate(model.shared_layers):
        # Current layer connects h_i -> h_{i+1}
        # We checked drift in layer i, which refers to h_{i+1} neurons.
        
        split_indices = drift_indices.get(i, []) # Indices in h_{i+1} to split
        k_split = len(split_indices)
        
        d_in = layer.in_features  # current h_i size (might have been expanded in prev iter?)
        d_out = layer.out_features # current h_{i+1} size
        
        new_d_out = d_out + k_split
        
        # We need a new layer [new_d_out, d_in] ?? 
        # Wait, if previous layer split neurons, d_in (h_i) also increased!
        # We need to track the *mapped* input size.
        
        pass 
        
    # This requires a more careful sequential construction.
    # Let's perform the split in-place or rebuild.
    
    # Simplified approach:
    # Apply splits layer by layer, updating the model state immediately so next layer sees correct input dim.
    
    sorted_layers = sorted(drift_indices.keys())
    
    # Note: If we split h1, shared_layers[0] output grows, shared_layers[1] input grows.
    # We must do this consistently.
    
    # We'll use a helper to expand specific layer output
    
    for layer_idx in sorted_layers:
         # Indices of neurons in OUTPUT of layer_idx to split
         indices_to_split = drift_indices[layer_idx]
         if not indices_to_split: continue
         
         # 1. Expand Output of shared_layers[layer_idx]
         layer = model.shared_layers[layer_idx]
         old_weight = layer.weight.data
         old_bias = layer.bias.data
         
         # Duplicate rows
         # New weights = old_weights + copies of split rows
         
         # Specifically:
         # Original rows: Keep as is (they are now the "newly adapted" version because we trained).
         # New rows (restored): We want the "old" weights from old_state_dict to preserve history?
         # "The first copy retains the weights of the old task, while the second copy adapts to the new task."
         # The "second copy" is what we have right now in `layer.weight` (adapted).
         # So we keep existing rows as the "second copy".
         # We add new rows which are restored from `old_state_dict` (first copy).
         
         # Getting old weights
         key = f"shared_layers.{layer_idx}.weight"
         old_w_orig = old_state_dict[key] # [old_out, old_in]
         key_b = f"shared_layers.{layer_idx}.bias"
         old_b_orig = old_state_dict[key_b]
         
         # Note: layer.weight might be larger than old_w_orig if we already expanded horizontally?
         # But usually Split happens before Expansion? Or parallel?
         # Assuming dimensions match roughly or we handle sub-regions.
         # For the neurons that existed (indices < old_w_orig.shape[0]), we can restore.
         
         rows_to_restore_w = []
         rows_to_restore_b = []
         
         for idx in indices_to_split:
             if idx < old_w_orig.shape[0]:
                 rows_to_restore_w.append(old_w_orig[idx])
                 rows_to_restore_b.append(old_b_orig[idx])
                 
         if not rows_to_restore_w:
             continue
             
         added_w = torch.stack(rows_to_restore_w).to(device)
         added_b = torch.stack(rows_to_restore_b).to(device)
         
         # Resize layer
         new_out = layer.out_features + len(rows_to_restore_w)
         new_layer = nn.Linear(layer.in_features, new_out).to(device)
         
         # Copy current weights to top
         new_layer.weight.data[:layer.out_features] = layer.weight.data
         new_layer.bias.data[:layer.out_features] = layer.bias.data
         
         # Append restored weights to bottom
         new_layer.weight.data[layer.out_features:] = added_w
         # We need to pad columns if input expanded? 
         # If layer.in_features > added_w.shape[1], we need to init the extra cols?
         # Ideally zero or copy current? 
         # Since these are "old" neurons, they shouldn't connect to "new" input neurons (which represent new features).
         # So zero padding for new columns is appropriate.
         current_in_dim = layer.in_features
         restored_in_dim = added_w.shape[1]
         if current_in_dim > restored_in_dim:
             # Look at new_layer.weight[bottom, restored_in_dim:]
             nn.init.constant_(new_layer.weight.data[layer.out_features:, restored_in_dim:], 0)
             
         new_layer.bias.data[layer.out_features:] = added_b
         
         model.shared_layers[layer_idx] = new_layer
         
         # Update hidden_dims
         # shared_layers[layer_idx] output corresponds to hidden_dims[layer_idx+1]
         model.hidden_dims[layer_idx+1] = new_out
         
         # 2. Update Input of Next Layer (shared_layers[layer_idx+1] or task_outputs)
         # The next layer must accept the new duplicated neurons.
         # The duplicated neurons (at the bottom) should have the same outgoing weights as the original neurons
         # so they propagate the signal similarly initially?
         # Or they should restore "old" outgoing weights? 
         # Paper: "The first copy retains the weights of the old task..."
         # This implies we should restore outgoing weights too?
         # But outgoing weights are in the NEXT layer.
         
         # Let's handle the next layer.
         if layer_idx < len(model.shared_layers) - 1:
             next_layer = model.shared_layers[layer_idx+1]
             # It is a Linear layer. We need to expand its columns (input dim).
             # We duplicate the columns corresponding to split indices.
             # Wait, the split indices were `indices_to_split`.
             # The new neurons are at indices `[old_out, old_out+1, ...]`.
             # Their behavior (outgoing) should match `indices_to_split`.
             # But should it match the "current" or "old" behavior?
             # If we want the restored neuron to act like the old one, we should restore outgoing weights too.
             
             # Fetch old op weights for next layer
             key_next = f"shared_layers.{layer_idx+1}.weight"
             old_w_next = old_state_dict.get(key_next)
             
             # Construct new next layer
             new_next = nn.Linear(new_out, next_layer.out_features).to(device)
             
             # Copy current weights (for the non-new input columns)
             new_next.weight.data[:, :next_layer.in_features] = next_layer.weight.data
             new_next.bias.data[:] = next_layer.bias.data # Bias doesn't change with input expansion
             
             # For the new columns (corresponding to restored neurons):
             # We should probably use the `old_w_next` columns corresponding to `indices_to_split`.
             # Assuming `old_w_next` exists and has those columns.
             
             start_col = next_layer.in_features
             for i, orig_idx in enumerate(indices_to_split):
                 if old_w_next is not None and orig_idx < old_w_next.shape[1]:
                     # Restore old outgoing weights
                     col = old_w_next[:, orig_idx]
                     new_next.weight.data[:, start_col+i] = col
                 else:
                     # Fallback: copy current outgoing weights (duplication)
                     new_next.weight.data[:, start_col+i] = next_layer.weight.data[:, orig_idx]
             
             model.shared_layers[layer_idx+1] = new_next
             
         else:
             # Next layer is Task Outputs
             # For EACH task output, we need to expand input dim.
             for tid, tail in model.task_outputs.items():
                 new_tail = nn.Linear(new_out, tail.out_features).to(device)
                 new_tail.weight.data[:, :tail.in_features] = tail.weight.data
                 new_tail.bias.data[:] = tail.bias.data
                 
                 # New columns
                 # Similar logic: restore old if possible? 
                 # But task_output for CURRENT task is new, so no old weights.
                 # For OLD tasks, we might want to restore?
                 # Since we only train current task, old tasks heads shouldn't have changed much?
                 # Assuming we want to preserve old task behavior:
                 # The "restored" neuron (at bottom) should connect to old task heads with OLD weights.
                 # The "adapted" neuron (original index) connects with CURRENT weights.
                 
                 # If we assume heads were frozen during this task training, old/current are same.
                 # So we basically duplicate the column.
                 
                 start_col = tail.in_features
                 for i, orig_idx in enumerate(indices_to_split):
                     new_tail.weight.data[:, start_col+i] = tail.weight.data[:, orig_idx]
                 
                 model.task_outputs[tid] = new_tail

    return model

