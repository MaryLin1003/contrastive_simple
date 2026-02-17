"""所有模型定义"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ========== 基础编码器 ==========
class BasicBlock(nn.Module):
    """基础残差块"""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetEncoder(nn.Module):
    """共享的ResNet编码器"""
    def __init__(self, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

# ========== 三个主要模型 ==========
class SupervisedModel(nn.Module):
    """监督学习模型"""
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features), features

class SimCLRModel(nn.Module):
    """SimCLR对比学习模型"""
    def __init__(self, projection_dim=128):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return z1, z2, h1, h2
    
    def encode(self, x):
        return self.encoder(x)

class MoCoModel(nn.Module):
    """MoCo动量对比模型"""
    def __init__(self, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        
        # 查询编码器
        self.encoder_q = ResNetEncoder()
        self.projector_q = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # 键编码器（动量更新）
        self.encoder_k = deepcopy(self.encoder_q)
        self.projector_k = deepcopy(self.projector_q)
        
        # 初始化键编码器参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # 创建队列
        self.register_buffer("queue", torch.randn(128, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新键编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的keys
        self.queue[:, ptr:ptr + batch_size] = keys.T
        
        # 移动指针
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """前向传播"""
        # 计算查询特征
        q = self.encoder_q(im_q)
        q = self.projector_q(q)
        q = nn.functional.normalize(q, dim=1)
        
        # 计算键特征
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = self.projector_k(k)
            k = nn.functional.normalize(k, dim=1)
        
        # 计算对比损失
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        return logits, labels, q, k
    
    def encode(self, x):
        return self.encoder_q(x)

# ========== 模型工厂 ==========
def create_model(model_name, config=None):
    """创建模型实例"""
    if model_name == 'supervised':
        return SupervisedModel()
    elif model_name == 'simclr':
        return SimCLRModel(projection_dim=config.get('projection_dim', 128) if config else 128)
    elif model_name == 'moco':
        return MoCoModel(
            K=config.get('queue_size', 65536) if config else 65536,
            m=config.get('momentum', 0.999) if config else 0.999,
            T=config.get('temperature', 0.07) if config else 0.07
        )
    else:
        raise ValueError(f"未知模型: {model_name}")