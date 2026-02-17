"""可视化脚本 - 纪占锋使用"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_features(models_dir='./results'):
    """加载所有模型的特征"""
    features = {}
    labels = None
    
    for model_name in ['supervised', 'simclr', 'moco']:
        feature_path = os.path.join(models_dir, model_name, 'test_features.npy')
        label_path = os.path.join(models_dir, model_name, 'test_labels.npy')
        
        if os.path.exists(feature_path):
            features[model_name] = np.load(feature_path)
            if labels is None:
                labels = np.load(label_path)
    
    return features, labels

def plot_tsne_comparison(features, labels, output_path='./figures/fig1_tsne.png'):
    """生成图1：t-SNE特征可视化"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    model_names = ['supervised', 'simclr', 'moco']
    titles = ['监督学习', 'SimCLR', 'MoCo v2']
    
    for idx, (model_name, title) in enumerate(zip(model_names, titles)):
        if model_name not in features:
            continue
            
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features[model_name][:1000])  # 只用1000个样本
        
        ax = axes[idx]
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                           c=labels[:1000], cmap='tab10', alpha=0.6, s=10)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('t-SNE维度2', fontsize=12)
        if idx == 1:
            ax.set_xlabel('t-SNE维度1', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图1已保存: {output_path}")

def plot_training_curves(models_dir='./results', output_path='./figures/fig2_training.png'):
    """生成图2：训练曲线"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = {'supervised': '#1f77b4', 'simclr': '#ff7f0e', 'moco': '#2ca02c'}
    labels = {'supervised': '监督学习', 'simclr': 'SimCLR', 'moco': 'MoCo v2'}
    
    for model_name in ['supervised', 'simclr', 'moco']:
        history_path = os.path.join(models_dir, model_name, 'training_history.json')
        
        if os.path.exists(history_path):
            import json
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # 损失曲线
            ax1.plot(history['train_loss'], color=colors[model_name], 
                    label=labels[model_name], linewidth=2)
            
            # 准确率曲线
            if 'train_acc' in history:
                ax2.plot(history['train_acc'], color=colors[model_name],
                        label=labels[model_name], linewidth=2)
    
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.set_title('(a) 训练损失收敛曲线', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('(b) 训练准确率变化', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图2已保存: {output_path}")

def generate_all_figures(models_dir='./results', output_dir='./figures'):
    """生成所有图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 开始生成可视化图表...")
    
    # 1. 加载特征
    features, labels = load_features(models_dir)
    
    if not features:
        print("⚠️  未找到特征文件，请先运行训练")
        return
    
    # 2. 生成图1
    plot_tsne_comparison(features, labels, 
                        os.path.join(output_dir, 'fig1_tsne.png'))
    
    # 3. 生成图2
    plot_training_curves(models_dir,
                        os.path.join(output_dir, 'fig2_training.png'))
    
    print(f"\n✨ 所有图表已保存到: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成可视化图表')
    parser.add_argument('--input', type=str, default='./results',
                       help='模型和特征目录')
    parser.add_argument('--output', type=str, default='./figures',
                       help='输出目录')
    
    args = parser.parse_args()
    generate_all_figures(args.input, args.output)