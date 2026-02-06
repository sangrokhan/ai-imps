import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

class SAGEAggregator(nn.Module):
    """
    GraphSAGE의 Aggregator (Mean, LSTM, Pooling 등) 중 Mean Aggregator 구현.
    """
    def __init__(self, in_features, out_features, dropout=0.0):
        super(SAGEAggregator, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, neighbor_x):
        # x: [N, in_features]
        # neighbor_x: [N, in_features] (neighbors' mean)
        combined = torch.cat([x, neighbor_x], dim=1)
        combined = self.dropout(combined)
        return self.linear(combined)

class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(SAGELayer, self).__init__()
        self.aggregator = SAGEAggregator(in_features, out_features, dropout)

    def forward(self, x, adj):
        # adj: [N, N] (degree-normalized adjacency matrix)
        neighbor_x = torch.mm(adj, x)
        h = self.aggregator(x, neighbor_x)
        return F.normalize(h, p=2, dim=1)

@MODEL_REGISTRY.register("graphsage")
class GraphSAGE(BaseModel):
    """
    Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    """
    def __init__(self, config):
        super(GraphSAGE, self).__init__(config)
        self.input_dim = config.get("input_dim", 1433)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.output_dim = config.get("output_dim", 7)
        self.dropout = config.get("dropout", 0.5)

        self.layer1 = SAGELayer(self.input_dim, self.hidden_dim, self.dropout)
        self.layer2 = SAGELayer(self.hidden_dim, self.output_dim, self.dropout)

    def forward(self, data):
        x, adj = data
        x = F.relu(self.layer1(x, adj))
        x = self.layer2(x, adj)
        return x

    def compute_loss(self, outputs, targets, mask=None):
        if mask is not None:
            return F.cross_entropy(outputs[mask], targets[mask])
        return F.cross_entropy(outputs, targets)
