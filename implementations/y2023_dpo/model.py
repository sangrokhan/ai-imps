import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY
import copy

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)

@MODEL_REGISTRY.register("DPO")
class DPOModel(BaseModel):
    """
    Direct Preference Optimization (DPO) Model Wrapper.
    Contains both the Policy (to be trained) and the Reference model (frozen).
    """
    def __init__(self, config):
        super().__init__(config)
        self.beta = config.get("beta", 0.1)
        
        in_dim = config.get("state_dim", 128)
        out_dim = config.get("action_dim", 10)
        
        # Policy Model
        self.policy = SimpleMLP(in_dim, out_dim)
        
        # Reference Model (Frozen)
        self.ref_model = copy.deepcopy(self.policy)
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.policy(x)

    def compute_loss(self, chosen_ids, rejected_ids, states):
        """
        DPO Loss: -log sigmoid( beta * (log(pi_chosen/ref_chosen) - log(pi_rejected/ref_rejected)) )
        
        Inputs:
            chosen_ids: [B] indices of preferred actions
            rejected_ids: [B] indices of dispreferred actions
            states: [B, state_dim] Context/State
        """
        # Get Logits
        policy_logits = self.policy(states)
        with torch.no_grad():
            ref_logits = self.ref_model(states)
            
        # Get Log Probs for Chosen and Rejected
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        policy_chosen_logps = policy_log_probs.gather(1, chosen_ids.unsqueeze(1)).squeeze(1)
        policy_rejected_logps = policy_log_probs.gather(1, rejected_ids.unsqueeze(1)).squeeze(1)
        
        ref_chosen_logps = ref_log_probs.gather(1, chosen_ids.unsqueeze(1)).squeeze(1)
        ref_rejected_logps = ref_log_probs.gather(1, rejected_ids.unsqueeze(1)).squeeze(1)
        
        # DPO Logic
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        losses = -F.logsigmoid(self.beta * logits)
        return losses.mean()
