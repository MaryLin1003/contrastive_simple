"""
工具函数模块
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

def setup_seed(seed: int = 42) -> None:
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ 随机种子已设置为: {seed}")

def get_device(use_cuda: bool = True) -> torch.device:
    """获取可用设备"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  使用CPU")
    
    return device

def save_checkpoint(
    state: Dict[str, Any],
    filename: str,
    is_best: bool = False,
    best_filename: str = None
) -> None:
    """保存模型检查点"""
    torch.save(state, filename)
    if is_best and best_filename:
        torch.save(state, best_filename)
        print(f"💾 最佳模型已保存: {best_filename}")

def load_checkpoint(
    filename: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """加载模型检查点"""
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(filename, map_location=device)
    
    # 加载模型参数
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 已加载检查点: {filename}")
    print(f"   轮次: {checkpoint.get('epoch', 'N/A')}")
    print(f"   损失: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint

def compute_accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """进度显示器"""
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置默认值
    defaults = {
        'data': {'num_workers': 2, 'train_split': 0.9, 'val_split': 0.1},
        'training': {'save_frequency': 10, 'save_best_only': True},
        'logging': {'tensorboard': False}
    }
    
    # 合并配置
    for section, default_values in defaults.items():
        if section in config:
            for key, value in default_values.items():
                config[section].setdefault(key, value)
        else:
            config[section] = default_values
    
    print(f"📋 配置文件已加载: {config_path}")
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """保存配置到YAML文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 配置已保存: {config_path}")

def save_training_history(history: Dict[str, Any], save_path: str) -> None:
    """保存训练历史"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 添加时间戳
    history['timestamp'] = datetime.now().isoformat()
    history['total_time'] = time.time() - history.get('start_time', time.time())
    
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"📊 训练历史已保存: {save_path}")

def print_config_summary(config: Dict[str, Any]) -> None:
    """打印配置摘要"""
    print("\n" + "="*50)
    print("📋 配置摘要")
    print("="*50)
    
    # 模型配置
    if 'model' in config:
        print(f"🤖 模型: {config['model'].get('name', '未知')}")
        if config['model'].get('name') == 'simclr':
            print(f"   投影维度: {config['model'].get('projection_dim', 128)}")
            print(f"   温度参数: {config['model'].get('temperature', 0.5)}")
        elif config['model'].get('name') == 'moco':
            print(f"   队列大小: {config['model'].get('queue_size', 65536)}")
            print(f"   动量参数: {config['model'].get('momentum', 0.999)}")
    
    # 数据配置
    if 'data' in config:
        print(f"📊 数据: {config['data'].get('name', '未知')}")
        print(f"   批大小: {config['data'].get('batch_size', 256)}")
        print(f"   数据线程: {config['data'].get('num_workers', 2)}")
    
    # 训练配置
    if 'training' in config:
        print(f"🏋️  训练轮次: {config['training'].get('epochs', 100)}")
        print(f"   学习率: {config['training'].get('learning_rate', 0.001)}")
        print(f"   优化器: {config['training'].get('optimizer', 'adam')}")
    
    print("="*50 + "\n")