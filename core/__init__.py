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
    CIFAR10Contrastive,
    
    # 数据加载函数
    get_dataloader,
    get_cifar10_loaders,
    get_transforms,
    
    # 数据增强
    ContrastiveTransformations,
    SimCLRAugmentation,
    MoCoAugmentation,
    
    # 数据工具
    download_cifar10,
    split_dataset,
    visualize_batch
)

# ========== 模型模块 ==========
from .models import (
    # 基础组件
    ResNetEncoder,
    BasicBlock,
    BottleneckBlock,
    
    # 主要模型
    SupervisedModel,
    SimCLRModel,
    MoCoModel,
    
    # 投影头
    MLPProjectionHead,
    LinearProjectionHead,
    
    # 模型工厂
    create_model,
    load_model,
    save_model,
    
    # 模型工具
    count_parameters,
    model_summary
)

# ========== 训练模块 ==========
from .trainers import (
    # 基础训练器
    BaseTrainer,
    
    # 特定训练器
    SupervisedTrainer,
    SimCLRTrainer,
    MoCoTrainer,
    
    # 训练工具
    setup_training,
    TrainingLogger,
    CheckpointManager
)

# ========== 损失函数模块 ==========
from .losses import (
    # 对比损失
    NTXentLoss,
    InfoNCELoss,
    ContrastiveLoss,
    
    # 分类损失
    CrossEntropyLoss,
    LabelSmoothingLoss,
    
    # 损失工具
    compute_contrastive_loss,
    compute_supervised_loss
)

# ========== 评估模块 ==========
from .evaluation import (
    # 评估器
    LinearEvaluator,
    KNNEvaluator,
    RetrievalEvaluator,
    
    # 评估指标
    compute_accuracy,
    compute_precision_recall,
    compute_f1_score,
    
    # 评估工具
    extract_features,
    evaluate_model,
    create_evaluation_report
)

# ========== 工具模块 ==========
from .utils import (
    # 设备与随机种子
    setup_seed,
    get_device,
    setup_device,
    
    # 配置管理
    load_config,
    save_config,
    merge_configs,
    
    # 检查点管理
    save_checkpoint,
    load_checkpoint,
    
    # 进度监控
    AverageMeter,
    ProgressMeter,
    TimeMeter,
    
    # 日志记录
    setup_logger,
    get_logger,
    
    # 可视化工具
    plot_training_curves,
    plot_confusion_matrix,
    plot_feature_distribution,
    
    # 其他工具
    dict_to_str,
    seconds_to_str,
    sizeof_fmt
)

# ========== 消融实验模块 ==========
from .ablation import (
    # 消融实验运行器
    AblationRunner,
    
    # 实验配置
    AblationConfig,
    ExperimentConfig,
    
    # 结果分析
    AblationAnalyzer,
    ResultAggregator,
    
    # 报告生成
    generate_ablation_report,
    plot_ablation_results
)

# ========== 导出所有公共接口 ==========
__all__ = [
    # 数据
    'CIFAR10Pair',
    'CIFAR10Contrastive',
    'get_dataloader',
    'get_cifar10_loaders',
    'get_transforms',
    'ContrastiveTransformations',
    'SimCLRAugmentation',
    'MoCoAugmentation',
    'download_cifar10',
    'split_dataset',
    'visualize_batch',
    
    # 模型
    'ResNetEncoder',
    'BasicBlock',
    'BottleneckBlock',
    'SupervisedModel',
    'SimCLRModel',
    'MoCoModel',
    'MLPProjectionHead',
    'LinearProjectionHead',
    'create_model',
    'load_model',
    'save_model',
    'count_parameters',
    'model_summary',
    
    # 训练
    'BaseTrainer',
    'SupervisedTrainer',
    'SimCLRTrainer',
    'MoCoTrainer',
    'setup_training',
    'TrainingLogger',
    'CheckpointManager',
    
    # 损失函数
    'NTXentLoss',
    'InfoNCELoss',
    'ContrastiveLoss',
    'CrossEntropyLoss',
    'LabelSmoothingLoss',
    'compute_contrastive_loss',
    'compute_supervised_loss',
    
    # 评估
    'LinearEvaluator',
    'KNNEvaluator',
    'RetrievalEvaluator',
    'compute_accuracy',
    'compute_precision_recall',
    'compute_f1_score',
    'extract_features',
    'evaluate_model',
    'create_evaluation_report',
    
    # 工具
    'setup_seed',
    'get_device',
    'setup_device',
    'load_config',
    'save_config',
    'merge_configs',
    'save_checkpoint',
    'load_checkpoint',
    'AverageMeter',
    'ProgressMeter',
    'TimeMeter',
    'setup_logger',
    'get_logger',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_feature_distribution',
    'dict_to_str',
    'seconds_to_str',
    'sizeof_fmt',
    
    # 消融实验
    'AblationRunner',
    'AblationConfig',
    'ExperimentConfig',
    'AblationAnalyzer',
    'ResultAggregator',
    'generate_ablation_report',
    'plot_ablation_results'
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
      • ablation - 消融实验
    
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