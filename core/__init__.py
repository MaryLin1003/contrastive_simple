%%writefile /kaggle/working/contrastive_simple/core/__init__.py
"""
对比学习项目核心模块
版本: 1.0.0
作者: 对比学习项目组 (林灵, 叶廷凯, 符武文, 纪占锋, 刘长蒙)
"""

__version__ = "1.0.0"
__author__ = "对比学习项目组"
__email__ = "contrastive-learning-group@example.com"
__description__ = "计算机视觉中对比学习的实现与比较研究"
__license__ = "MIT"

# ========== 数据模块 ==========
from .data import (
    # 数据集类
    CIFAR10Pair,
    
    # 数据加载函数
    get_dataloader,
    
    # 数据工具
    download_cifar10,
)

# ========== 模型模块 ==========
from .models import (
    # 基础组件
    ResNetEncoder,
    BasicBlock,
    
    # 主要模型
    SupervisedModel,
    SimCLRModel,
    MoCoModel,
    
    # 模型工厂
    create_model,
    
    # 模型工具
    count_parameters,
)

# ========== 训练模块 ==========
from .trainers import (
    # 基础训练器
    BaseTrainer,
    
    # 训练工具
    TrainingLogger,
    CheckpointManager
)

# ========== 损失函数模块 ==========
from .losses import (
    # 对比损失
    NTXentLoss,
    
    # 分类损失
    CrossEntropyLoss,
)

# ========== 评估模块 ==========
from .evaluation import (
    # 评估器
    LinearEvaluator,
    KNNEvaluator,
    
    # 评估指标
    compute_accuracy,
    
    # 评估工具
    extract_features,
)

# ========== 工具模块 ==========
from .utils import (
    # 设备与随机种子
    setup_seed,
    get_device,
    
    # 配置管理
    load_config,
    save_config,
    
    # 检查点管理
    save_checkpoint,
    load_checkpoint,
    
    # 进度监控
    AverageMeter,
    ProgressMeter,
)

# ========== 导出所有公共接口 ==========
__all__ = [
    # 数据
    'CIFAR10Pair',
    'get_dataloader',
    'download_cifar10',
    
    # 模型
    'ResNetEncoder',
    'BasicBlock',
    'SupervisedModel',
    'SimCLRModel',
    'MoCoModel',
    'create_model',
    'count_parameters',
    
    # 训练
    'BaseTrainer',
    'TrainingLogger',
    'CheckpointManager',
    
    # 损失函数
    'NTXentLoss',
    'CrossEntropyLoss',
    
    # 评估
    'LinearEvaluator',
    'KNNEvaluator',
    'compute_accuracy',
    'extract_features',
    
    # 工具
    'setup_seed',
    'get_device',
    'load_config',
    'save_config',
    'save_checkpoint',
    'load_checkpoint',
    'AverageMeter',
    'ProgressMeter',
]

# ========== 初始化信息 ==========
def print_info():
    """打印模块信息"""
    info = f"""
    ============================================
    对比学习项目核心模块 v{__version__}
    ============================================
    作者: {__author__}
    描述: {__description__}
    许可证: {__license__}
    
    可用模块:
      • data     - 数据加载与增强
      • models   - 模型定义 (监督/SimCLR/MoCo)
      • trainers - 训练器与训练循环
      • losses   - 损失函数
      • evaluation - 评估与指标
      • utils    - 工具函数
    
    使用示例:
      >>> from core import create_model, get_dataloader
      >>> model = create_model('simclr')
      >>> dataloader = get_dataloader(config)
    ============================================
    """
    print(info)

# 导入时自动打印信息（仅第一次）
if 'PRINTED_INFO' not in globals():
    PRINTED_INFO = True
    print_info()
