#!/usr/bin/env python3
"""
极简数据下载脚本
不依赖torchvision，纯Python实现
"""

import os
import sys
import tarfile
import urllib.request
import pickle
import numpy as np
from pathlib import Path
import ssl

# 绕过SSL验证（解决某些网络问题）
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, filename):
    """下载文件并显示进度"""
    print(f"📥 下载: {url}")
    print(f"保存到: {filename}")
    
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r进度: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, reporthook=progress)
        print(f"\n✅ 下载完成")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def extract_tar(tar_path, extract_to):
    """解压tar.gz文件"""
    print(f"📦 解压: {tar_path}")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        print(f"✅ 解压完成到: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False

def verify_cifar10(data_dir):
    """验证CIFAR-10数据集"""
    print("\n🔍 验证数据集...")
    
    required_files = [
        'batches.meta',
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
        'test_batch'
    ]
    
    data_dir = Path(data_dir) / 'cifar-10-batches-py'
    
    if not data_dir.exists():
        print(f"❌ 数据集目录不存在: {data_dir}")
        return False
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺失文件: {missing_files}")
        return False
    
    print(f"✅ 数据集完整!")
    
    # 显示文件信息
    total_size = 0
    for file in required_files:
        file_path = data_dir / file
        size = file_path.stat().st_size
        total_size += size
        print(f"   {file}: {size/1024/1024:.1f} MB")
    
    print(f"📊 总大小: {total_size/1024/1024:.1f} MB")
    
    # 尝试读取一个文件验证数据格式
    try:
        with open(data_dir / 'data_batch_1', 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            print(f"📊 数据批次1信息:")
            print(f"   数据形状: {dict[b'data'].shape}")
            print(f"   标签数量: {len(dict[b'labels'])}")
    except Exception as e:
        print(f"⚠️  数据读取测试失败: {e}")
    
    return True

def main():
    print("="*60)
    print("CIFAR-10 数据集下载工具 (极简版)")
    print("="*60)
    
    # 参数
    data_root = "./data"
    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    # 备用镜像（如果主链接失败）
    mirrors = [
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "http://pjreddie.com/media/files/cifar-10-python.tar.gz",
    ]
    
    # 创建数据目录
    data_dir = Path(data_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = data_dir / "cifar-10-python.tar.gz"
    extract_dir = data_dir / "cifar-10-batches-py"
    
    # 检查是否已存在
    if extract_dir.exists():
        print(f"📁 数据集似乎已存在: {extract_dir}")
        if verify_cifar10(data_root):
            print("\n🎉 数据集已就绪，无需下载")
            return
    
    # 下载数据集
    print(f"目标目录: {data_dir.absolute()}")
    
    success = False
    for mirror in mirrors:
        print(f"\n尝试镜像: {mirror}")
        if download_file(mirror, tar_path):
            success = True
            break
        else:
            print(f"镜像失败，尝试下一个...")
    
    if not success:
        print("\n❌ 所有镜像下载失败")
        print("\n💡 手动下载方法:")
        print("1. 访问: https://www.cs.toronto.edu/~kriz/cifar.html")
        print("2. 下载 'cifar-10-python.tar.gz' (约163MB)")
        print("3. 保存到: ./data/cifar-10-python.tar.gz")
        print("4. 重新运行此脚本")
        return
    
    # 解压
    if not extract_tar(tar_path, data_dir):
        print("❌ 解压失败")
        return
    
    # 验证
    if verify_cifar10(data_root):
        print("\n🎉 CIFAR-10数据集准备完成!")
        
        # 可选：删除压缩包节省空间
        delete = input("\n是否删除压缩包以节省空间？(y/n): ").lower()
        if delete == 'y':
            tar_path.unlink()
            print(f"🗑️  已删除: {tar_path}")
    else:
        print("\n❌ 数据集验证失败")

if __name__ == "__main__":
    main()