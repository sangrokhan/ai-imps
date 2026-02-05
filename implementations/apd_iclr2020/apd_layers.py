
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class APDLayer(nn.Module):
    def __init__(self):
        super(APDLayer, self).__init__()

    def get_effective_weight(self, task_id):
        raise NotImplementedError

class APDLinear(APDLayer):
    def __init__(self, in_features, out_features, bias=True, initial_task_id=0):
        super(APDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Shared parameters
        self.weight_shared = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Task specific parameters storage
        self.weight_task = nn.ParameterDict()
        self.task_masks_shared = nn.ParameterDict() # Mask (Ms)
        self.task_masks_task = nn.ParameterDict()   # Mask (Mt)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_shared, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_shared)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id):
        task_id = str(task_id)
        if task_id in self.weight_task:
            return
            
        # Initialize task specific weight with small random values or zeros
        self.weight_task[task_id] = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        nn.init.zeros_(self.weight_task[task_id]) # Start with zero difference
        
        # Initialize masks
        # Assuming masks are learnable and continuous for now to allow gradient descent optimization
        # Sigmoid will be applied during forward to constrain to [0,1] or simple Softmax if needed, 
        # but paper often uses learnable mask directly or with gate. 
        # Here we initialize them to be "all open" or "learnable".
        # Let's initialize them such that initially it relies mostly on shared weights.
        self.task_masks_shared[task_id] = nn.Parameter(torch.ones(self.out_features, self.in_features))
        self.task_masks_task[task_id] = nn.Parameter(torch.ones(self.out_features, self.in_features))

    def forward(self, input, task_id):
        task_id = str(task_id)
        if task_id not in self.weight_task:
             # Basic handling if task not added, though caller should ensure it is added.
             # Fallback to just shared weights equivalent
             return F.linear(input, self.weight_shared, self.bias)

        # Formula: y = (W_shared * M_s + W_task[t] * M_t) x + b
        # We assume elemental-wise multiplication for masking
        
        ws = self.weight_shared # Shared weight
        wt = self.weight_task[task_id] # Task specific weight
        ms = self.task_masks_shared[task_id] # Shared mask
        mt = self.task_masks_task[task_id] # Task mask
        
        # Effective weight
        effective_weight = ws * ms + wt * mt
        
        return F.linear(input, effective_weight, self.bias)

class APDConv2d(APDLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(APDConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Kernel shape helper
        if isinstance(kernel_size, int):
            self.ks = (kernel_size, kernel_size)
        else:
            self.ks = kernel_size
            
        # Shared parameters
        self.weight_shared = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.ks))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Task specific parameters storage
        self.weight_task = nn.ParameterDict()
        self.task_masks_shared = nn.ParameterDict()
        self.task_masks_task = nn.ParameterDict()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_shared, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_shared)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def add_task(self, task_id):
        task_id = str(task_id)
        if task_id in self.weight_task:
            return
            
        weight_shape = (self.out_channels, self.in_channels // self.groups, *self.ks)
        
        # Task weight
        self.weight_task[task_id] = nn.Parameter(torch.Tensor(*weight_shape))
        nn.init.zeros_(self.weight_task[task_id])
        
        # Masks
        self.task_masks_shared[task_id] = nn.Parameter(torch.ones(*weight_shape))
        self.task_masks_task[task_id] = nn.Parameter(torch.ones(*weight_shape))

    def forward(self, input, task_id):
        task_id = str(task_id)
        if task_id not in self.weight_task:
            return F.conv2d(input, self.weight_shared, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        ws = self.weight_shared
        wt = self.weight_task[task_id]
        ms = self.task_masks_shared[task_id]
        mt = self.task_masks_task[task_id]
        
        effective_weight = ws * ms + wt * mt
        
        return F.conv2d(input, effective_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
