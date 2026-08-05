"""
棱镜模型 - 共享配置
所有脚本的统一参数定义
"""
from pathlib import Path

# ===================== 路径 =====================
BASE_DIR = Path(__file__).parent
SHAPES_DIR = BASE_DIR / "shapes"
MAPS_DIR = BASE_DIR / "maps"
CHECKPOINT_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# ===================== 模型参数 =====================
IMG_SIZE = 64               # 工作分辨率
CHANNELS_OUT = 2            # 输出通道 (dx, dy)
K_LAYERS = 8                # 多层深度编码层数
COND_CHANNELS = K_LAYERS + 2 + 2  # 深度(8)+厚度(1)+掩码(1)+valid(1)+confidence(1)=12
HIDDEN_DIM = 64             # U-Net 基础通道

# ===================== 扩散参数 =====================
T_STEPS = 1000              # 扩散步数
BETA_START = 1e-4
BETA_END = 0.02
GUIDANCE_STRENGTH = 0.7     # 引导采样强度

# ===================== 训练参数 =====================
LEARNING_RATE = 2e-4
DEFAULT_EPOCHS = 1500
DEFAULT_BATCH = 8
RANDOM_MASK_RATIO = 0.15    # 随机掩码增强比例
SEED = 42

# ===================== 数据划分 =====================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ===================== 物理参数 =====================
N_REFRACT = 1.43            # 玻璃折射率
OUTPUT_Z = -0.5             # 出射记录平面

# ===================== Python 环境（可选，按需修改） =====================
# PYTHON_ENV = r"path/to/your/python.exe"  # 仅 download_models.py 等脚本使用
