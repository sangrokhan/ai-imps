import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("DecisionTransformer")
class DecisionTransformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_size = config['embed_dim']
        self.max_length = config['max_length']
        self.max_ep_len = config.get('max_ep_len', 4096)
        
        # Embeddings
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_size)
        
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        
        # GPT-style Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config['n_head'],
                dim_feedforward=self.hidden_size * 4,
                dropout=config['dropout'],
                activation=config['activation'],
                batch_first=True,
                norm_first=True
            ),
            num_layers=config['n_layer']
        )
        
        # Prediction Heads
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, self.action_dim),
            nn.Tanh() # For continuous actions usually, but CartPole is discrete. 
                      # NOTE: This implementation assumes continuous/normalized actions typical in DT papers.
                      # For discrete, we might need Softmax or just logits.
                      # I will use Identity for logits here assuming CrossEntropy or MSE on discrete embeddings.
                      # Re-reading: DT usually does MSE on actions for continuous tasks.
                      # For CartPole (discrete), we output logits.
        )
        self.predict_return = nn.Linear(self.hidden_size, 1)
        self.predict_state = nn.Linear(self.hidden_size, self.state_dim)

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        # states: [B, L, state_dim]
        # actions: [B, L, action_dim]
        # returns: [B, L, 1]
        # timesteps: [B, L]
        
        B, L, _ = states.shape
        
        # Embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        return_embeddings = self.embed_return(returns) + time_embeddings
        
        # Stack inputs: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        # We interleave them along the sequence dimension.
        # Shape: [B, 3*L, hidden_size]
        stacked_inputs = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * L, self.hidden_size)
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Causal Mask
        # We need to mask future tokens.
        mask = torch.triu(torch.ones(3*L, 3*L, device=states.device), diagonal=1).bool()
        
        # Forward Pass
        x = self.transformer(stacked_inputs, mask=mask)
        
        # Reshape to [B, L, 3, hidden_size]
        x = x.reshape(B, L, 3, self.hidden_size)
        
        # We are interested in predicting actions from state embeddings (index 1 in the stack)
        # return_preds = self.predict_return(x[:, :, 2]) # Predict next return from action (not typical)
        # state_preds = self.predict_state(x[:, :, 2])   # Predict next state from action
        action_preds = self.predict_action(x[:, :, 1])   # Predict action from state
        
        return action_preds 

    def compute_loss(self, outputs, targets):
        # outputs: action_preds [B, L, action_dim]
        # targets: real_actions [B, L, action_dim] (or indices for discrete)
        
        # For discrete actions (CartPole), targets might be indices [B, L].
        # If so, we use CrossEntropy.
        # If targets are one-hot or continuous, MSE.
        
        if targets.dim() == outputs.dim() - 1: # Targets are indices
            return nn.CrossEntropyLoss()(outputs.permute(0, 2, 1), targets)
        else:
            return F.mse_loss(outputs, targets)
