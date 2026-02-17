"""评估脚本 - 刘长蒙使用"""
import argparse
import os
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from core.data import get_dataloader
from core.models import create_model

def linear_evaluation(model_path, output_dir):
    """线性评估预训练模型"""
    # 1. 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # 2. 创建模型
    model = create_model(config['model']['name'], config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 获取数据
    train_loader = get_dataloader(config, train=True)
    test_loader = get_dataloader(config, train=False)
    
    # 4. 提取特征
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for x1, x2, label in loader:
                if config['model']['name'] == 'supervised':
                    _, feat = model(x1)
                else:
                    feat = model.encode(x1)
                features.append(feat.numpy())
                labels.append(label.numpy())
        return np.concatenate(features), np.concatenate(labels)
    
    print("📊 提取特征...")
    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)
    
    # 5. 训练线性分类器
    print("🔧 训练线性分类器...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_features, train_labels)
    
    # 6. 评估
    train_pred = clf.predict(train_features)
    test_pred = clf.predict(test_features)
    
    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    
    # 7. 保存结果
    results = {
        'model': config['model']['name'],
        'linear_train_accuracy': float(train_acc),
        'linear_test_accuracy': float(test_acc),
        'num_train_samples': len(train_labels),
        'num_test_samples': len(test_labels),
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_accuracy': float(checkpoint.get('acc', 0))
    }
    
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{config['model']['name']}_linear_eval.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n📈 线性评估结果:")
    print(f"  训练准确率: {train_acc:.2%}")
    print(f"  测试准确率: {test_acc:.2%}")
    print(f"  结果保存到: {result_file}")
    
    return results

def evaluate_all(models_dir='./results', output_dir='./tables'):
    """评估所有模型并生成表格"""
    os.makedirs(output_dir, exist_ok=True)
    
    models = ['supervised', 'simclr', 'moco']
    all_results = {}
    
    for model_name in models:
        model_path = os.path.join(models_dir, model_name, 'model_best.pt')
        if os.path.exists(model_path):
            print(f"\n🔍 评估 {model_name}...")
            results = linear_evaluation(model_path, output_dir)
            all_results[model_name] = results
        else:
            print(f"⚠️  未找到 {model_name} 模型")
    
    # 生成表格
    generate_table(all_results, output_dir)
    
    return all_results

def generate_table(results, output_dir):
    """生成性能对比表格"""
    import pandas as pd
    
    table_data = []
    for model_name, res in results.items():
        table_data.append({
            '方法': {'supervised': '监督学习', 'simclr': 'SimCLR', 'moco': 'MoCo v2'}[model_name],
            '线性评估准确率 (%)': f"{res['linear_test_accuracy']*100:.1f}",
            '训练样本数': res['num_train_samples'],
            '测试样本数': res['num_test_samples']
        })
    
    df = pd.DataFrame(table_data)
    
    # 保存为多种格式
    df.to_csv(os.path.join(output_dir, 'table1_performance.csv'), index=False)
    df.to_markdown(os.path.join(output_dir, 'table1_performance.md'), index=False)
    
    print("\n📋 表格已生成:")
    print(df.to_string(index=False))
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--model', type=str, choices=['supervised', 'simclr', 'moco', 'all'],
                       default='all', help='要评估的模型')
    parser.add_argument('--input', type=str, default='./results',
                       help='模型目录')
    parser.add_argument('--output', type=str, default='./tables',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        evaluate_all(args.input, args.output)
    else:
        model_path = os.path.join(args.input, args.model, 'model_best.pt')
        linear_evaluation(model_path, args.output)