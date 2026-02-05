import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel

class APDModel(BaseModel):
    def __init__(self, config):
        super(APDModel, self).__init__(config)
        self.input_dim = config.get("input_dim", 784)
        self.hidden_dim = config.get("hidden_dim", 400)
        self.output_dim = config.get("output_dim", 10)
        self.num_tasks = config.get("num_tasks", 10)
        self.W_shared = nn.ParameterDict({
            'fc1': nn.Parameter(torch.randn(self.hidden_dim, self.input_dim)),
            'fc2': nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim)),
            'fc3': nn.Parameter(torch.randn(self.output_dim, self.hidden_dim))
        })
        self.alphas = nn.Parameter(torch.ones(self.num_tasks, 3))
        self.masks = nn.ParameterDict()
        for t in range(self.num_tasks):
            self.masks[f'task_{t}_m1'] = nn.Parameter(torch.ones(self.hidden_dim, self.input_dim))
            self.masks[f'task_{t}_m2'] = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))
            self.masks[f'task_{t}_m3'] = nn.Parameter(torch.ones(self.output_dim, self.hidden_dim))

    def get_task_weight(self, task_id, layer_name):
        layer_idx = {'fc1': 0, 'fc2': 1, 'fc3': 2}[layer_name]
        alpha = self.alphas[task_id, layer_idx]
        mask = self.masks[f'task_{task_id}_{layer_name.replace("fc", "m")}']
        shared_w = self.W_shared[layer_name]
        return alpha * (shared_w * mask)

    def forward(self, x, task_id=0):
        x = x.view(x.size(0), -1)
        w1 = self.get_task_weight(task_id, 'fc1')
        x = F.linear(x, w1)
        x = F.relu(x)
        w2 = self.get_task_weight(task_id, 'fc2')
        x = F.linear(x, w2)
        x = F.relu(x)
        w3 = self.get_task_weight(task_id, 'fc3')
        x = F.linear(x, w3)
        return x

    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
