import os
import yaml
import json
import time
import torch
import torch.nn as nn
from core.data import get_dataloader
from core.trainers import BaseTrainer

# 强制使用GPU
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

def load_full_config(config_path):
    """完整加载配置文件，处理include继承"""
    
    with open(config_path, 'r') as f:
        current_config = yaml.safe_load(f)
    
    if 'include' in current_config:
        # 计算基础配置的路径
        base_path = os.path.join(os.path.dirname(config_path), current_config['include'])
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        print(f"📂 加载基础配置: {current_config['include']}")
        
        # 深度合并配置
        full_config = base_config.copy()
        
        # 合并各个部分
        for section in ['model', 'data', 'training', 'evaluation', 'logging']:
            if section in current_config:
                if section not in full_config:
                    full_config[section] = {}
                for key, value in current_config[section].items():
                    full_config[section][key] = value
        
        # 合并experiment部分
        if 'experiment' in current_config:
            full_config['experiment'] = current_config['experiment']
        
        return full_config
    
    return current_config

def run_ablation_experiment(exp_name, config_path, target_epochs, output_base):
    """运行单个消融实验"""
    
    print(f"\n{'='*60}")
    print(f"🔬 消融实验: {exp_name}")
    print(f"{'='*60}")
    
    # 创建输出目录
    output_dir = os.path.join(output_base, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否已有进度
    history_path = os.path.join(output_dir, 'training_history.json')
    start_epoch = 1
    history = {'train_loss': []}
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        start_epoch = len(history['train_loss']) + 1
        print(f"📂 找到已有进度，从第 {start_epoch} 轮继续")
    
    # 加载配置
    config = load_full_config(config_path)
    # 设置目标轮次
    config['training']['epochs'] = target_epochs
    
    print(f"\n📋 实验信息:")
    print(f"  实验名称: {exp_name}")
    print(f"  模型: {config['model']['name']}")
    print(f"  批次大小: {config['data']['batch_size']}")
    print(f"  目标轮次: {target_epochs}")
    print(f"  已训练: {start_epoch - 1} 轮")
    print(f"  设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # 准备数据
    print(f"\n📊 加载数据...")
    train_loader = get_dataloader(config, train=True)
    test_loader = get_dataloader(config, train=False)
    
    # 设置训练器
    trainer = BaseTrainer(config)
    model = trainer.setup()
    
    # 训练循环
    print(f"\n🚀 开始训练...")
    start_time = time.time()
    
    for epoch in range(start_epoch, target_epochs + 1):
        epoch_start = time.time()
        
        # 训练一个epoch
        loss, _ = trainer.train_epoch(train_loader, epoch)
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        history['train_loss'].append(float(loss))
        
        print(f"Epoch {epoch:3d}/{target_epochs} | Loss: {loss:.4f} | Time: {epoch_time:.1f}s")
        
        # 每20轮保存一次检查点
        if epoch % 20 == 0 or epoch == target_epochs:
            # 保存模型
            checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
            trainer.save_checkpoint(epoch, loss, 0.0, checkpoint_path)
            
            # 保存训练历史
            history['total_time'] = time.time() - start_time
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"  💾 已保存检查点 (epoch {epoch})")
    
    # 保存最终特征
    print(f"\n💾 保存特征...")
    trainer.save_features(test_loader, os.path.join(output_dir, 'test'))
    
    print(f"\n✅ {exp_name} 完成！结果保存在: {output_dir}")
    print(f"⏱️  总训练时间: {history['total_time']/3600:.2f} 小时")
    
    return history

def check_progress():
    """检查实验进度"""
    base_dirs = ['./results/ablation_simplified/epochs', 
                 './results/ablation_simplified/augmentation']
    
    print("\n📊 消融实验进度")
    print("="*60)
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        
        exp_type = os.path.basename(base_dir)
        print(f"\n📁 {exp_type}:")
        
        for exp_name in os.listdir(base_dir):
            exp_path = os.path.join(base_dir, exp_name)
            history_path = os.path.join(exp_path, 'training_history.json')
            
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                completed = len(history.get('train_loss', []))
                
                # 根据实验名确定总轮次
                if '50' in exp_name:
                    total = 50
                elif '200' in exp_name:
                    total = 200
                else:
                    total = 100
                
                percentage = (completed / total) * 100
                bar = '█' * int(percentage/10) + '░' * (10 - int(percentage/10))
                print(f"  {exp_name:12}: [{bar}] {completed}/{total} 轮 ({percentage:.1f}%)")
                if completed > 0:
                    print(f"             最新损失: {history['train_loss'][-1]:.4f}")
            else:
                print(f"  {exp_name:12}: 未开始")

# ========== 直接运行实验 ==========
print("\n" + "="*60)
print("🚀 开始运行简化版消融实验")
print("="*60)

# 检查当前进度
check_progress()

# 询问是否继续
print("\n是否开始/继续实验？")
print("1. 运行所有实验")
print("2. 只运行预训练轮次实验")
print("3. 只运行数据增强实验")
print("4. 检查进度")

choice = input("请输入选择 (1-4): ").strip()

results = {}

if choice == '1':
    # ========== 预训练轮次 ==========
    print("\n📁 预训练轮次实验")
    print("-"*40)
    
    print("\n🔹 实验: epochs_50")
    history_50 = run_ablation_experiment(
        exp_name='epochs_50',
        config_path='configs/ablation/epochs/epochs_50.yaml',
        target_epochs=50,
        output_base='./results/ablation_simplified/epochs'
    )
    results['epochs_50'] = {
        'final_loss': history_50['train_loss'][-1],
        'time': history_50['total_time'] / 3600
    }
    
    print("\n🔹 实验: epochs_200")
    history_200 = run_ablation_experiment(
        exp_name='epochs_200',
        config_path='configs/ablation/epochs/epochs_200.yaml',
        target_epochs=200,
        output_base='./results/ablation_simplified/epochs'
    )
    results['epochs_200'] = {
        'final_loss': history_200['train_loss'][-1],
        'time': history_200['total_time'] / 3600
    }
    
    # ========== 数据增强 ==========
    print("\n📁 数据增强实验")
    print("-"*40)
    
    print("\n🔹 实验: aug_basic")
    history_basic = run_ablation_experiment(
        exp_name='aug_basic',
        config_path='configs/ablation/augmentation/basic.yaml',
        target_epochs=100,
        output_base='./results/ablation_simplified/augmentation'
    )
    results['aug_basic'] = {
        'final_loss': history_basic['train_loss'][-1],
        'time': history_basic['total_time'] / 3600
    }
    
    print("\n🔹 实验: aug_full")
    history_full = run_ablation_experiment(
        exp_name='aug_full',
        config_path='configs/ablation/augmentation/full.yaml',
        target_epochs=100,
        output_base='./results/ablation_simplified/augmentation'
    )
    results['aug_full'] = {
        'final_loss': history_full['train_loss'][-1],
        'time': history_full['total_time'] / 3600
    }

elif choice == '2':
    # 只跑预训练轮次
    print("\n📁 预训练轮次实验")
    print("-"*40)
    
    print("\n🔹 实验: epochs_50")
    history_50 = run_ablation_experiment(
        exp_name='epochs_50',
        config_path='configs/ablation/epochs/epochs_50.yaml',
        target_epochs=50,
        output_base='./results/ablation_simplified/epochs'
    )
    results['epochs_50'] = {
        'final_loss': history_50['train_loss'][-1],
        'time': history_50['total_time'] / 3600
    }
    
    print("\n🔹 实验: epochs_200")
    history_200 = run_ablation_experiment(
        exp_name='epochs_200',
        config_path='configs/ablation/epochs/epochs_200.yaml',
        target_epochs=200,
        output_base='./results/ablation_simplified/epochs'
    )
    results['epochs_200'] = {
        'final_loss': history_200['train_loss'][-1],
        'time': history_200['total_time'] / 3600
    }

elif choice == '3':
    # 只跑数据增强
    print("\n📁 数据增强实验")
    print("-"*40)
    
    print("\n🔹 实验: aug_basic")
    history_basic = run_ablation_experiment(
        exp_name='aug_basic',
        config_path='configs/ablation/augmentation/basic.yaml',
        target_epochs=100,
        output_base='./results/ablation_simplified/augmentation'
    )
    results['aug_basic'] = {
        'final_loss': history_basic['train_loss'][-1],
        'time': history_basic['total_time'] / 3600
    }
    
    print("\n🔹 实验: aug_full")
    history_full = run_ablation_experiment(
        exp_name='aug_full',
        config_path='configs/ablation/augmentation/full.yaml',
        target_epochs=100,
        output_base='./results/ablation_simplified/augmentation'
    )
    results['aug_full'] = {
        'final_loss': history_full['train_loss'][-1],
        'time': history_full['total_time'] / 3600
    }

elif choice == '4':
    check_progress()
    exit()

else:
    print("输入错误")
    exit()

# ========== 打印结果汇总 ==========
if results:
    print("\n" + "="*60)
    print("📊 简化版消融实验结果汇总")
    print("="*60)
    
    if 'epochs_50' in results:
        print("\n1. 预训练轮次影响:")
        print(f"   50轮:  损失={results['epochs_50']['final_loss']:.4f}, 时间={results['epochs_50']['time']:.2f}h")
        print(f"   200轮: 损失={results['epochs_200']['final_loss']:.4f}, 时间={results['epochs_200']['time']:.2f}h")
        print(f"   改进:  {results['epochs_50']['final_loss']-results['epochs_200']['final_loss']:.4f}")
    
    if 'aug_basic' in results:
        print("\n2. 数据增强影响:")
        print(f"   基础增强: 损失={results['aug_basic']['final_loss']:.4f}")
        print(f"   完整增强: 损失={results['aug_full']['final_loss']:.4f}")
        print(f"   改进:     {results['aug_basic']['final_loss']-results['aug_full']['final_loss']:.4f}")
    
    # 保存结果到文件
    results_path = './results/ablation_simplified/summary.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 结果已保存到: {results_path}")
