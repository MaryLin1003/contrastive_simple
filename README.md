# 消融实验配置

这个目录包含用于消融实验的配置文件，用于生成论文中的表2-5。

## 文件结构
ablation/
├── epochs/ # 表2：预训练轮次影响
│ ├── epochs_50.yaml
│ ├── epochs_100.yaml
│ ├── epochs_150.yaml
│ └── epochs_200.yaml
│
├── augmentation/ # 表3：数据增强影响
│ ├── basic.yaml # 基础增强
│ ├── no_color.yaml # 无颜色抖动
│ ├── no_blur.yaml # 无高斯模糊
│ └── full.yaml # 完整增强
│
├── projection/ # 表4：投影头维度影响
│ ├── dim_64.yaml
│ ├── dim_128.yaml
│ ├── dim_256.yaml
│ └── no_proj.yaml # 无投影头
│
└── batch_size/ # 表5：批次大小影响
├── bs_64.yaml
├── bs_128.yaml
├── bs_256.yaml
└── bs_512.yaml


## 使用方法

1. **运行单个消融实验**：
```bash
python train.py --config configs/ablation/epochs/epochs_50.yaml
2. **批量运行所有消融实验**:
python scripts/run_ablation.py --experiment epochs
3. **生成消融实验表格**：
python evaluate.py --ablation --input ./results/ablation --output ./tables


实验设计
表2：预训练轮次 (epochs)
50, 100, 150, 200轮

固定其他所有参数

观察收敛速度和最终性能

表3：数据增强 (augmentation)
Basic: 随机裁剪 + 水平翻转

+Color: 基础 + 颜色抖动

+Blur: 基础 + 高斯模糊

Full: 所有增强组合

表4：投影头维度 (projection)
64, 128, 256维

无投影头（直接使用编码器特征）

观察维度对对比学习的影响

表5：批次大小 (batch_size)
64, 128, 256, 512

观察批次大小对对比学习的影响

记录内存使用和训练时间