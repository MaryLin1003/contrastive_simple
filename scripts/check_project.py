#!/usr/bin/env python3
"""
项目就绪检查脚本
"""

import os
import sys
from pathlib import Path

def check_directory_structure():
    """检查目录结构"""
    print("📁 检查目录结构...")
    
    required_dirs = [
        'configs',
        'configs/ablation',
        'configs/ablation/epochs',
        'configs/ablation/augmentation', 
        'configs/ablation/projection',
        'configs/ablation/batch_size',
        'core',
        'scripts',
        'data',
        'results'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ 缺失目录: {missing_dirs}")
        return False
    else:
        print("✅ 目录结构完整")
        return True

def check_required_files():
    """检查必需文件"""
    print("\n📄 检查核心文件...")
    
    required_files = [
        'train.py',
        'evaluate.py', 
        'visualize.py',
        'requirements.txt',
        'configs/supervised.yaml',
        'configs/simclr.yaml',
        'configs/moco.yaml',
        'core/__init__.py',
        'core/data.py',
        'core/models.py',
        'core/trainers.py',
        'core/utils.py',
        'scripts/download_data_simple.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺失文件: {missing_files}")
        return False
    else:
        print("✅ 核心文件完整")
        return True

def check_data():
    """检查数据"""
    print("\n📊 检查数据...")
    
    data_dir = Path('./data/cifar-10-batches-py')
    required_data_files = [
        'batches.meta',
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
        'test_batch'
    ]
    
    if not data_dir.exists():
        print("❌ 数据目录不存在")
        return False
    
    missing_files = []
    for file in required_data_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺失数据文件: {missing_files}")
        return False
    
    # 检查文件大小
    total_size = 0
    for file in required_data_files:
        file_path = data_dir / file
        size = file_path.stat().st_size
        total_size += size
    
    print(f"✅ 数据文件完整")
    print(f"   数据目录: {data_dir}")
    print(f"   总大小: {total_size/1024/1024:.1f} MB")
    
    return True

def check_dependencies():
    """检查依赖"""
    print("\n🔧 检查Python依赖...")
    
    required_modules = ['numpy', 'torch', 'torchvision', 'matplotlib', 'sklearn']
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"⚠️  缺失模块: {missing_modules}")
        print(f"   运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 核心依赖已安装")
        return True

def quick_test():
    """快速测试"""
    print("\n🧪 快速功能测试...")
    
    try:
        # 测试导入核心模块
        print("  测试导入核心模块...")
        sys.path.insert(0, '.')
        from core.data import CIFAR10Pair
        from core.models import create_model
        from core.utils import setup_seed, get_device
        print("  ✅ 核心模块导入成功")
        
        # 测试配置读取
        print("  测试配置读取...")
        import yaml
        with open('configs/supervised.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✅ 配置读取成功: {config['model']['name']}")
        
        # 测试数据加载（小样本）
        print("  测试数据加载...")
        import torch
        from torch.utils.data import DataLoader
        
        # 使用简化的数据加载进行测试
        dataset = CIFAR10Pair(root='./data', train=True)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        batch = next(iter(dataloader))
        print(f"  ✅ 数据加载成功")
        print(f"     批次大小: {len(batch)}")
        print(f"     图像形状: {batch[0][0].shape}")
        
        # 测试模型创建
        print("  测试模型创建...")
        device = get_device(use_cuda=False)  # 测试用CPU
        model = create_model('supervised')
        model = model.to(device)
        
        print(f"  ✅ 模型创建成功")
        print(f"     模型类型: {type(model).__name__}")
        print(f"     参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("项目就绪验证检查")
    print("="*60)
    
    checks = [
        ("目录结构", check_directory_structure),
        ("核心文件", check_required_files),
        ("数据集", check_data),
        ("Python依赖", check_dependencies),
        ("快速功能测试", quick_test)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} 检查异常: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 项目完全就绪！可以开始训练！")
        print("\n下一步：")
        print("1. 运行监督学习测试：python train.py --model supervised --epochs 5")
        print("2. 如果测试成功，可以开始正式训练")
    else:
        print("⚠️  项目存在一些问题，请先解决")
    
    print("="*60)

if __name__ == "__main__":
    main()