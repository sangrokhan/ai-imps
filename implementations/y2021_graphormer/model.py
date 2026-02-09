import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

class GraphormerLayer(nn.Module):
    """
    Graphormer의 핵심인 Structural Encoding이 적용된 Transformer Layer의 간소화 버전.
    Centrality encoding 및 Spatial encoding 개념을 포함함.
    """
    def __init__(self, d_model, n_heads, dim_feedforward=512, dropout=0.1):
        super(GraphormerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, spatial_bias=None, padding_mask=None):
        # x: [B, N, d_model]
        # spatial_bias: [B, N, N] (SPD-based bias)
        
        # PyTorch MultiheadAttention warns when key_padding_mask (bool) and attn_mask (float) are mixed.
        # Since spatial_bias is a float additive bias, we convert padding_mask to float if it's bool.
        if spatial_bias is not None and padding_mask is not None:
            if padding_mask.dtype == torch.bool and not spatial_bias.dtype == torch.bool:
                new_padding_mask = torch.zeros_like(padding_mask, dtype=spatial_bias.dtype)
                padding_mask = new_padding_mask.masked_fill(padding_mask, float("-inf"))

        # Self-attention with Spatial Bias
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask, attn_mask=spatial_bias)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # FFN
        ffn_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

@MODEL_REGISTRY.register("graphormer")
class Graphormer(BaseModel):
    """
    Do Transformers Really Perform Bad for Graph Representation? (NeurIPS 2021)
    """
    def __init__(self, config):
        super(Graphormer, self).__init__(config)
        self.d_model = config.get("d_model", 64)
        self.n_heads = config.get("n_heads", 8)
        self.n_layers = config.get("n_layers", 6)
        self.input_dim = config.get("input_dim", 1433)
        self.output_dim = config.get("output_dim", 7)
        self.max_degree = config.get("max_degree", 128)

        # Centrality Encoding
        self.in_degree_encoder = nn.Embedding(self.max_degree, self.d_model)
        self.out_degree_encoder = nn.Embedding(self.max_degree, self.d_model)
        
        self.node_encoder = nn.Linear(self.input_dim, self.d_model)
        
        self.layers = nn.ModuleList([
            GraphormerLayer(self.d_model, self.n_heads)
            for _ in range(self.n_layers)
        ])
        
        self.classifier = nn.Linear(self.d_model, self.output_dim)

    def forward(self, data):
        """
        data: (x, in_degree, out_degree, spatial_bias, padding_mask)
        """
        x, in_degree, out_degree, spatial_bias, padding_mask = data
        
        # Node Features + Centrality Encoding
        h = self.node_encoder(x) + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        
        for layer in self.layers:
            h = layer(h, spatial_bias, padding_mask)
            
        return self.classifier(h)

    def compute_loss(self, outputs, targets, mask=None):
        if mask is not None:
            return F.cross_entropy(outputs[mask], targets[mask])
        return F.cross_entropy(outputs, targets)
