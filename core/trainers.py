
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np

# 导入模型创建函数
from .models import create_model

class BaseTrainer:
    """基础训练器 - 支持监督学习、SimCLR、MoCo"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()  # 统一使用交叉熵损失
        
    def setup(self):
        """设置模型、优化器"""
        model_name = self.config['model']['name']
        self.model = create_model(model_name, self.config['model']).to(self.device)
        
        # 优化器
        lr = self.config['training']['learning_rate']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training'].get('weight_decay', 1e-4)
        )
        
        return self.model
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch - 根据模型类型选择不同训练逻辑"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            model_name = self.config['model']['name']
            
            if model_name == 'supervised':
                loss, acc, batch_size = self._train_supervised_batch(batch)
                
            elif model_name == 'simclr':
                loss, acc, batch_size = self._train_simclr_batch(batch)
                
            elif model_name == 'moco':
                loss, acc, batch_size = self._train_moco_batch(batch)
                
            else:
                raise ValueError(f"未知模型类型: {model_name}")
            
            total_loss += loss * batch_size
            total_acc += acc * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{acc:.2%}'
            })
        
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc
    
    def _train_supervised_batch(self, batch):
        """监督学习单批次训练"""
        x1, x2, labels = batch
        x1, labels = x1.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()
        logits, _ = self.model(x1)
        loss = self.criterion(logits, labels)
        
        # 计算准确率
        _, predicted = logits.max(1)
        acc = (predicted == labels).float().mean().item()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc, x1.size(0)
    
    def _train_simclr_batch(self, batch):
        """SimCLR单批次训练 - 修正版损失函数"""
        x1, x2, _ = batch
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        self.optimizer.zero_grad()
        z1, z2, _, _ = self.model(x1, x2)
        
        batch_size = z1.shape[0]
        
        # SimCLR损失实现
        # 1. L2归一化
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # 2. 拼接特征
        features = torch.cat([z1, z2], dim=0)  # [2*batch_size, dim]
        
        # 3. 计算相似度矩阵
        similarity = torch.matmul(features, features.T)  # [2*batch_size, 2*batch_size]
        
        # 4. 创建标签 - 修正版本
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            labels[i] = i + batch_size - 1  # 第一个view的正样本
            labels[i + batch_size] = i      # 第二个view的正样本
        
        # 5. 移除对角线（自身对比）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity = similarity[~mask].view(2 * batch_size, -1)
        
        # 6. 应用温度参数
        temperature = self.config['model'].get('temperature', 0.5)
        similarity /= temperature
        
        # 7. 计算损失
        loss = self.criterion(similarity, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), 0.0, batch_size  # SimCLR无准确率概念
    
    def _train_moco_batch(self, batch):
        """MoCo单批次训练"""
        x1, x2, _ = batch
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        self.optimizer.zero_grad()
        logits, labels, _, _ = self.model(x1, x2)
        
        loss = self.criterion(logits, labels)
        
        # 计算对比准确率
        _, predicted = logits.max(1)
        acc = (predicted == labels).float().mean().item()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc, x1.size(0)
    
    def save_checkpoint(self, epoch, loss, acc, path):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
            'config': self.config
        }, path)
        print(f"💾 检查点已保存: {path}")
    
    def save_features(self, test_loader, save_path):
        """保存特征用于可视化"""
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                x1, x2, labels = batch
                x1 = x1.to(self.device)
                
                if self.config['model']['name'] == 'supervised':
                    _, features = self.model(x1)
                else:
                    features = self.model.encode(x1)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        np.save(save_path + '_features.npy', all_features)
        np.save(save_path + '_labels.npy', all_labels)
        
        print(f"✅ 特征已保存到: {save_path}_features.npy")
        print(f"   特征形状: {all_features.shape}, 标签形状: {all_labels.shape}")
