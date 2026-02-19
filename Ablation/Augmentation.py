import os
import sys
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# ========== 0. 设置数据集路径 ==========
print("="*60)
print("🔗 设置CIFAR-10数据集路径")
print("="*60)

data_path = '/kaggle/input/datasets/pankrzysiu/cifar10-python'
print(f"✅ 数据集路径: {data_path}")
!ls -la {data_path}/cifar-10-batches-py/ | head -10

# ========== 1. 数据加载类 ==========
class CIFAR10Pair(torch.utils.data.Dataset):
    """生成对比学习所需的图像对"""
    def __init__(self, root=None, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        self.dataset = datasets.CIFAR10(
            root=self.root, 
            train=train, 
            download=False, 
            transform=None
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            return self.transform(img), self.transform(img), label
        return img, img, label

def get_dataloader(config, train=True):
    """获取数据加载器"""
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    
    dataset = CIFAR10Pair(
        root='/kaggle/input/datasets/pankrzysiu/cifar10-python',
        train=train,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=train,
        num_workers=0,
        pin_memory=True,
        drop_last=train
    )

# ========== 2. 模型定义 ==========
class BasicBlock(nn.Module):
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
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

class SimCLRModel(nn.Module):
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

def create_model(model_name, model_config):
    if model_name == 'simclr':
        return SimCLRModel(projection_dim=model_config.get('projection_dim', 128))
    else:
        raise ValueError(f"未知模型: {model_name}")

# ========== 3. 训练器类 ==========
class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
    def setup(self):
        model_name = self.config['model']['name']
        self.model = create_model(model_name, self.config['model']).to(self.device)
        
        lr = self.config['training']['learning_rate']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training'].get('weight_decay', 1e-4)
        )
        return self.model
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x1, x2, _ = batch
            x1, x2 = x1.to(self.device), x2.to(self.device)
            
            self.optimizer.zero_grad()
            z1, z2, _, _ = self.model(x1, x2)
            
            batch_size = z1.shape[0]
            
            # SimCLR损失
            z1 = nn.functional.normalize(z1, dim=1)
            z2 = nn.functional.normalize(z2, dim=1)
            
            features = torch.cat([z1, z2], dim=0)
            similarity = torch.matmul(features, features.T)
            
            # 创建标签
            labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
            for i in range(batch_size):
                labels[i] = i + batch_size - 1
                labels[i + batch_size] = i
            
            # 移除对角线
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
            similarity = similarity[~mask].view(2 * batch_size, -1)
            
            # 温度参数
            temperature = self.config['model'].get('temperature', 0.5)
            similarity /= temperature
            
            loss = self.criterion(similarity, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}: loss={loss.item():.4f}")
        
        return total_loss / total_samples, 0.0
    
    def save_checkpoint(self, epoch, loss, acc, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
            'config': self.config
        }, path)
    
    def save_features(self, test_loader, save_path):
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                x1, x2, labels = batch
                x1 = x1.to(self.device)
                features = self.model.encode(x1)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        np.save(save_path + '_features.npy', all_features)
        np.save(save_path + '_labels.npy', all_labels)
        print(f"✅ 特征已保存到: {save_path}_features.npy")

# ========== 4. 配置加载函数 ==========
def load_full_config(config_path):
    with open(config_path, 'r') as f:
        current = yaml.safe_load(f)
    
    if 'include' in current:
        base_path = os.path.join(os.path.dirname(config_path), current['include'])
        with open(base_path, 'r') as f:
            base = yaml.safe_load(f)
        
        full = base.copy()
        for key, value in current.items():
            if key != 'include':
                if isinstance(value, dict) and key in full:
                    full[key].update(value)
                else:
                    full[key] = value
        return full
    return current

# ========== 5. 实验运行函数 ==========
def run_ablation_experiment(exp_name, config_path, target_epochs, output_base):
    print(f"\n{'='*60}")
    print(f"🔬 消融实验: {exp_name}")
    print(f"{'='*60}")
    
    output_dir = os.path.join(output_base, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    history_path = os.path.join(output_dir, 'training_history.json')
    start_epoch = 1
    history = {'train_loss': []}
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        start_epoch = len(history['train_loss']) + 1
        print(f"📂 找到已有进度，从第 {start_epoch} 轮继续")
    
    config = load_full_config(config_path)
    config['training']['epochs'] = target_epochs
    
    print(f"\n📋 实验信息:")
    print(f"  实验名称: {exp_name}")
    print(f"  模型: {config['model']['name']}")
    print(f"  批次大小: {config['data']['batch_size']}")
    print(f"  目标轮次: {target_epochs}")
    print(f"  已训练: {start_epoch - 1} 轮")
    print(f"  设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    train_loader = get_dataloader(config, train=True)
    test_loader = get_dataloader(config, train=False)
    
    trainer = BaseTrainer(config)
    model = trainer.setup()
    
    print(f"\n🚀 开始训练...")
    start_time = time.time()
    
    for epoch in range(start_epoch, target_epochs + 1):
        epoch_start = time.time()
        loss, _ = trainer.train_epoch(train_loader, epoch)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(float(loss))
        print(f"Epoch {epoch:3d}/{target_epochs} | Loss: {loss:.4f} | Time: {epoch_time:.1f}s")
        
        if epoch % 20 == 0 or epoch == target_epochs:
            checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
            trainer.save_checkpoint(epoch, loss, 0.0, checkpoint_path)
            
            history['total_time'] = time.time() - start_time
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"  💾 已保存检查点 (epoch {epoch})")
    
    print(f"\n💾 保存特征...")
    trainer.save_features(test_loader, os.path.join(output_dir, 'test'))
    
    print(f"\n✅ {exp_name} 完成！")
    print(f"⏱️  总训练时间: {history['total_time']/3600:.2f} 小时")
    
    return history

# ========== 6. 主程序：运行数据增强实验 ==========
print("="*60)
print("🔍 环境检查")
print("="*60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")

print("\n" + "="*60)
print("🚀 开始运行数据增强实验")
print("="*60)

# 运行实验
results = {}

# 先测试5轮，确保一切正常
print("\n🔹 测试实验: aug_basic (5轮)")
history_basic = run_ablation_experiment(
    exp_name='aug_basic_test',
    config_path='configs/ablation/augmentation/basic.yaml',
    target_epochs=5,
    output_base='./results/ablation_simplified/augmentation'
)
results['aug_basic_test'] = {
    'final_loss': history_basic['train_loss'][-1],
    'time': history_basic['total_time'] / 3600
}

print("\n✅ 测试完成！可以开始正式训练")
print("如需正式训练，请修改 target_epochs=100 后重新运行")
