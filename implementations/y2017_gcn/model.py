import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

class GCNLayer(nn.Module):
    """
    Kipf & Welling (2017)의 Graph Convolutional Layer 구현.
    H^(l+1) = \sigma(D'^-1/2 A' D'^-1/2 H^l W^l)
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: [N, in_features]
        # adj: [N, N] (normalized with self-loops)
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output

@MODEL_REGISTRY.register("gcn")
class GCN(BaseModel):
    """
    Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    """
    def __init__(self, config):
        super(GCN, self).__init__(config)
        self.input_dim = config.get("input_dim", 1433) # Cora dataset default
        self.hidden_dim = config.get("hidden_dim", 16)
        self.output_dim = config.get("output_dim", 7)
        self.dropout = config.get("dropout", 0.5)

        self.gc1 = GCNLayer(self.input_dim, self.hidden_dim)
        self.gc2 = GCNLayer(self.hidden_dim, self.output_dim)

    def forward(self, data):
        """
        data: (x, adj) 튜플을 기대함
        x: [N, input_dim]
        adj: [N, N] (adjacency matrix)
        """
        x, adj = data
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def compute_loss(self, outputs, targets, mask=None):
        """
        outputs: [N, output_dim]
        targets: [N]
        mask: 훈련/검증용 노드 마스크
        """
        if mask is not None:
            return F.cross_entropy(outputs[mask], targets[mask])
        return F.cross_entropy(outputs, targets)
