import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

class GATLayer(nn.Module):
    """
    Veličković et al. (2018)의 Graph Attention Layer 구현.
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [N, in_features]
        # adj: [N, N] (adjacency matrix with self-loops)
        Wh = torch.mm(h, self.W) # [N, out_features]
        N = Wh.size(0)

        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # [N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) # [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

@MODEL_REGISTRY.register("gat")
class GAT(BaseModel):
    """
    Graph Attention Networks (ICLR 2018)
    """
    def __init__(self, config):
        super(GAT, self).__init__(config)
        self.input_dim = config.get("input_dim", 1433)
        self.hidden_dim = config.get("hidden_dim", 8)
        self.output_dim = config.get("output_dim", 7)
        self.dropout = config.get("dropout", 0.6)
        self.alpha = config.get("alpha", 0.2)
        self.n_heads = config.get("n_heads", 8)

        self.attentions = nn.ModuleList([
            GATLayer(self.input_dim, self.hidden_dim, dropout=self.dropout, alpha=self.alpha, concat=True)
            for _ in range(self.n_heads)
        ])
        self.out_att = GATLayer(self.hidden_dim * self.n_heads, self.output_dim, dropout=self.dropout, alpha=self.alpha, concat=False)

    def forward(self, data):
        x, adj = data
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

    def compute_loss(self, outputs, targets, mask=None):
        if mask is not None:
            return F.cross_entropy(outputs[mask], targets[mask])
        return F.cross_entropy(outputs, targets)
