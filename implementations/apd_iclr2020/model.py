import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY
from .apd_layers import APDConv2d, APDLinear

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = APDConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = APDConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_layer = APDConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)
            self.has_shortcut_layer = True
        else:
            self.has_shortcut_layer = False

    def add_task(self, task_id):
        self.conv1.add_task(task_id)
        self.conv2.add_task(task_id)
        if self.has_shortcut_layer:
            self.shortcut_layer.add_task(task_id)

    def forward(self, x, task_id):
        out = F.relu(self.bn1(self.conv1(x, task_id)))
        out = self.bn2(self.conv2(out, task_id))
        
        shortcut = x
        if self.has_shortcut_layer:
            shortcut = self.shortcut_bn(self.shortcut_layer(x, task_id))
            
        out += shortcut
        out = F.relu(out)
        return out

@MODEL_REGISTRY.register("apd_resnet18")
class ResNet18APD(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        num_classes = config.get('num_classes', 100)
        self.in_planes = 64
        
        self.conv1 = APDConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.linear = APDLinear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.ModuleList(layers)

    def add_task(self, task_id):
        self.conv1.add_task(task_id)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.add_task(task_id)
        self.linear.add_task(task_id)

    def forward(self, x, **kwargs):
        task_id = kwargs.get('task_id', 0)
        out = F.relu(self.bn1(self.conv1(x, task_id)))
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                out = block(out, task_id)
                
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out, task_id)
        return out
