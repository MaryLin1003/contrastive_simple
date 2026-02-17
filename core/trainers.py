"""训练器"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os

class BaseTrainer:
    """基础训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def setup(self):
        """设置模型、优化器、损失函数"""
        model_name = self.config['model']['name']
        self.model = create_model(model_name, self.config['model']).to(self.device)
        
        # 优化器
        lr = self.config['training']['learning_rate']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config['training'].get('weight_decay', 1e-4)
        )
        
        # 损失函数
        if model_name == 'supervised':
            self.criterion = nn.CrossEntropyLoss()
        elif model_name in ['simclr', 'moco']:
            self.criterion = nn.CrossEntropyLoss()
        
        return self.model
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            if self.config['model']['name'] == 'supervised':
                (x1, x2, labels) = batch
                x1, labels = x1.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model(x1)
                loss = self.criterion(logits, labels)
                
                # 计算准确率
                _, predicted = logits.max(1)
                acc = (predicted == labels).float().mean()
                
            elif self.config['model']['name'] == 'simclr':
                (x1, x2, _) = batch
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                self.optimizer.zero_grad()
                z1, z2, _, _ = self.model(x1, x2)
                
                # SimCLR损失
                batch_size = z1.shape[0]
                features = torch.cat([z1, z2], dim=0)
                similarity_matrix = torch.matmul(features, features.T)
                
                # 创建标签
                labels = torch.cat([torch.arange(batch_size) + batch_size, 
                                  torch.arange(batch_size)], dim=0).to(self.device)
                mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
                similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
                
                # 应用温度
                temperature = self.config['model'].get('temperature', 0.5)
                similarity_matrix /= temperature
                
                loss = self.criterion(similarity_matrix, labels)
                acc = torch.tensor(0.0)  # 对比学习无准确率概念
            
            elif self.config['model']['name'] == 'moco':
                (x1, x2, _) = batch
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                self.optimizer.zero_grad()
                logits, labels, _, _ = self.model(x1, x2)
                loss = self.criterion(logits, labels)
                
                # 计算对比准确率
                _, predicted = logits.max(1)
                acc = (predicted == labels).float().mean()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x1.size(0)
            total_acc += acc.item() * x1.size(0)
            total_samples += x1.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.2%}'
            })
        
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc
    
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
    
    def save_features(self, test_loader, save_path):
        """保存特征用于可视化"""
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if self.config['model']['name'] == 'supervised':
                    (x1, x2, labels) = batch
                    x1 = x1.to(self.device)
                    _, features = self.model(x1)
                else:
                    (x1, x2, labels) = batch
                    x1 = x1.to(self.device)
                    features = self.model.encode(x1)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        import numpy as np
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        np.save(save_path + '_features.npy', all_features)
        np.save(save_path + '_labels.npy', all_labels)
        
        print(f"✅ 特征已保存到: {save_path}_features.npy")