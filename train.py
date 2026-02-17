"""主训练脚本 - 运行这个训练模型！"""
import argparse
import yaml
import os
import time
import json
from core.data import get_dataloader
from core.trainers import BaseTrainer

def train(config_path, output_dir):
    """训练模型"""
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🚀 开始训练: {config['model']['name']}")
    print(f"📁 输出目录: {output_dir}")
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # 4. 准备数据
    print("📊 加载数据...")
    train_loader = get_dataloader(config, train=True)
    test_loader = get_dataloader(config, train=False)
    
    # 5. 设置训练器
    trainer = BaseTrainer(config)
    model = trainer.setup()
    
    print(f"📈 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 设备: {trainer.device}")
    print(f"📅 总轮次: {config['training']['epochs']}")
    print("-" * 50)
    
    # 6. 训练循环
    epochs = config['training']['epochs']
    best_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'epoch_time': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # 训练一个epoch
        loss, acc = trainer.train_epoch(train_loader, epoch)
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        history['epoch_time'].append(epoch_time)
        
        print(f"✅ Epoch {epoch:3d}/{epochs} | "
              f"Loss: {loss:.4f} | Acc: {acc:.2%} | "
              f"Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            trainer.save_checkpoint(
                epoch, loss, acc,
                os.path.join(output_dir, 'model_best.pt')
            )
        
        # 定期保存
        if epoch % 10 == 0 or epoch == epochs:
            trainer.save_checkpoint(
                epoch, loss, acc,
                os.path.join(output_dir, f'model_epoch_{epoch}.pt')
            )
    
    total_time = time.time() - start_time
    
    # 7. 保存特征用于可视化
    print("\n💾 保存特征用于可视化...")
    trainer.save_features(
        test_loader,
        os.path.join(output_dir, 'test')
    )
    
    # 8. 保存训练历史
    history['total_time'] = total_time
    history['best_acc'] = best_acc
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 9. 打印总结
    print("\n" + "="*50)
    print(f"🎉 训练完成!")
    print(f"⏱️  总时间: {total_time/3600:.2f} 小时")
    print(f"🏆 最佳准确率: {best_acc:.2%}")
    print(f"📁 结果保存在: {output_dir}")
    print("="*50)
    
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练对比学习模型')
    parser.add_argument('--model', type=str, required=True,
                       choices=['supervised', 'simclr', 'moco'],
                       help='要训练的模型')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选）')
    parser.add_argument('--output', type=str, default='./results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 自动选择配置文件
    if args.config is None:
        args.config = f'configs/{args.model}.yaml'
    
    # 设置输出目录
    output_dir = os.path.join(args.output, args.model)
    
    # 运行训练
    train(args.config, output_dir)