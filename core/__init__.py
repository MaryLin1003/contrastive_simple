"""
对比学习项目核心模块
版本: 1.0.0
作者: 对比学习项目组 (林灵, 叶廷凯, 符武文, 纪占锋, 刘长蒙)
"""

__version__ = "1.0.0"
__author__ = "对比学习项目组"
__description__ = "计算机视觉中对比学习的实现与比较研究"
__license__ = "MIT"

# ========== 数据模块 ==========
from .data import (
    CIFAR10Pair,
    get_dataloader,
)

# ========== 模型模块 ==========
from .models import (
    ResNetEncoder,
    BasicBlock,
    SupervisedModel,
    SimCLRModel,
    MoCoModel,
    create_model,
)

# ========== 训练模块 ==========
from .trainers import (
    BaseTrainer,
)

# ========== 工具模块 ==========
from .utils import (
    setup_seed,
    get_device,
    load_config,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    ProgressMeter,
)

# ========== 导出所有公共接口 ==========
__all__ = [
    # 数据
    'CIFAR10Pair',
    'get_dataloader',
    
    # 模型
    'ResNetEncoder',
    'BasicBlock',
    'SupervisedModel',
    'SimCLRModel',
    'MoCoModel',
    'create_model',
    
    # 训练
    'BaseTrainer',
    
    # 工具
    'setup_seed',
    'get_device',
    'load_config',
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
      • trainers - 训练器
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
