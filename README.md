# 棱镜模型 (Lens Model)

基于**条件扩散模型**的 3D 棱镜光路映射预测系统。输入任意三角网格模型，模型预测平行光穿过棱镜后的出射光路位移场（2D→2D 映射），可用于反向光路设计、光学仿真可视化等场景。

## 项目原理

```
平行光 (z=+∞, 方向 0,0,-1)
   ↓ ↓ ↓ ↓ ↓
┌─────────────────┐  ← 入射平面
│   空气 (n=1.0)   │
│  ┌───────────┐  │  ← 玻璃棱镜 (n=1.43)
│  │ Snell 折射 │  │
│  │   内部传播  │  │
│  └───────────┘  │
│   空气 (n=1.0)   │
└─────────────────┘  ← 出射记录平面 (z=-0.5)
```

- **输入**：3D 网格的多层深度编码（Layered Depth Encoding，最多 8 层穿透表面 + 累计厚度 + 占据掩码）
- **输出**：归一化 2D 位移场 `(H, W, 2)` = 出射坐标 − 入射坐标
- **模型**：U-Net 条件扩散模型（DDPM），支持 DDIM 加速采样和引导式补全

## 目录结构

```
棱镜模型/
├── config.py              # 共享配置（路径、模型/训练/物理参数）
├── requirements.txt       # Python 依赖
│
├── train_diffusion.py     # 扩散模型训练主脚本
├── inference_diffusion.py # 单形状推理（DDIM 采样）
├── evaluate_diffusion.py  # 模型评估与指标计算
├── visualize_diffusion.py # 结果可视化
│
├── generate_shapes.py     # 3D 形状批量生成（PLY 格式）
├── shape2map.py           # 光路映射物理计算（Snell 折射追踪）
├── download_models.py     # 开源模型下载
├── organize_modelnet.py   # ModelNet 数据集整理
│
├── shapes/                # 3D 模型文件（01-10 类，PLY 格式）
│   ├── 01_正多面体/       # 正四面体、正六面体、正八面体、正十二面体、正二十面体
│   ├── 02_凸多面体/       # 截角四面体、立方八面体等
│   ├── 03_球体与锥体/     # 球体、圆锥、棱锥等
│   ├── 04_多亏格曲面/     # 环面(亏格1/2/3)、克莱因瓶、莫比乌斯带
│   ├── 05_动物/           # 兔子、猫、鸟、鱼、鹿、乌龟
│   ├── 06_植物/           # 阔叶树、松树、仙人掌、花、蘑菇
│   ├── 07_工具/           # 钳子、锤子、斧头、螺丝刀、扳手、手锯
│   ├── 08_建筑/           # 房屋、城堡、塔楼、拱桥、金字塔
│   ├── 09_人物动作/       # 站立、行走、奔跑、坐姿、挥手
│   └── 10_其他/           # 心形、小船、飞机、汽车、椅子、桌子等
│
├── maps/                  # 光路映射数据（与 shapes 对应，.npz + .txt）
├── figures/               # 可视化对比图（PNG）
├── results/               # 模型预测输出（.npz）
├── models/                # 训练好的模型权重 + 数据划分
│   ├── best_model.pt
│   ├── final_model.pt
│   └── data_split.json
│
└── .gitignore
```

## 环境搭建

### 1. 克隆仓库

```bash
git clone git@github.com:Darkhe55/LensModel.git
cd LensModel
```

### 2. 创建 Python 环境

推荐 Python 3.10+，使用 Conda 或 venv：

```bash
# Conda
conda create -n lensmodel python=3.10
conda activate lensmodel

# 或 venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：
- `torch >= 2.0.0` — 深度学习框架
- `numpy >= 1.24.0` — 数值计算
- `scipy >= 1.10.0` — 科学计算
- `matplotlib >= 3.7.0` — 可视化

可选依赖：
- `trimesh >= 3.20.0` — 3D 模型加载加速

### 4. GPU 支持（推荐）

训练推荐使用 CUDA GPU。安装 PyTorch CUDA 版本：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 使用方式

### 第一步：生成 3D 形状（可选）

如果 shapes/ 目录为空或需要重新生成：

```bash
python generate_shapes.py
```

在 `shapes/` 下生成 01-10 共 10 个类别的 PLY 模型文件。

### 第二步：计算光路映射

对每个 3D 模型，用 Snell 折射定律追踪光路，生成训练数据：

```bash
# 处理所有形状（128×128 分辨率）
python shape2map.py

# 高分辨率
python shape2map.py --resolution 256

# 只处理特定文件夹
python shape2map.py --folder "01_正多面体"

# 只处理单个模型
python shape2map.py --model "shapes/01_正多面体/正四面体.ply"

# 自定义折射率（默认 1.43 为玻璃）
python shape2map.py --n_refract 1.52

# 跳过已存在的 map（增量处理）
python shape2map.py --skip_existing

# 限制发散距离和路径长度（复杂模型）
python shape2map.py --max_diverge 30 --max_path 100

# 多线程加速
python shape2map.py --workers 8
```

生成的 `.npz` 文件包含：
| 字段 | 说明 |
|------|------|
| `input_grid` | 入射光线坐标 `(res, res, 2)` |
| `output_grid` | 出射光线坐标 `(res, res, 2)` |
| `valid_mask` | 有效光线布尔掩码 |
| `inverse_map` | 逆映射：出射→入射 |
| `bbox_min/max` | 模型包围盒 |

### 第三步：训练扩散模型

```bash
# 完整流程（训练 + 采样 + 评估）
python train_diffusion.py

# 仅训练
python train_diffusion.py --mode train

# 自定义训练参数
python train_diffusion.py --mode train --epochs 2000 --batch_size 4

# 调整 KL 散度权重（控制潜在空间正则化）
python train_diffusion.py --mode train --kl_weight 0.01   # 加强正则化
python train_diffusion.py --mode train --kl_weight 0.0    # 纯 MSE，关闭 KL

# 仅采样（从已训练模型生成预测）
python train_diffusion.py --mode sample

# 仅评估
python train_diffusion.py --mode evaluate
```

模型权重保存在 `models/best_model.pt`（验证集最优）和 `models/final_model.pt`（最终）。

### 第四步：推理预测

对任意 3D 模型预测光路映射：

```bash
# 基本推理
python inference_diffusion.py shapes/01_正多面体/正四面体.ply

# 指定输出路径
python inference_diffusion.py shapes/05_动物/兔子.ply --output results/兔子_pred.npz

# DDIM 加速（默认 50 步，比 DDPM 快 10-20 倍）
python inference_diffusion.py shapes/04_多亏格曲面/环面_亏格1.ply --ddim_steps 50
```

### 第五步：评估与可视化

```bash
# 评估模型在测试集上的表现
python evaluate_diffusion.py

# 生成对比图、误差直方图、预测场可视化
python visualize_diffusion.py
```

结果保存在 `figures/`（PNG 图表）和 `results/`（NPZ 数据）。

## 关键参数速查

在 `config.py` 中集中管理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IMG_SIZE` | 64 | 工作分辨率 |
| `K_LAYERS` | 8 | 多层深度编码层数 |
| `HIDDEN_DIM` | 64 | U-Net 基础通道数 |
| `T_STEPS` | 1000 | 扩散总步数 |
| `LEARNING_RATE` | 2e-4 | 学习率 |
| `DEFAULT_EPOCHS` | 1500 | 默认训练轮数 |
| `GUIDANCE_STRENGTH` | 0.7 | 引导采样强度 |
| `N_REFRACT` | 1.43 | 玻璃折射率 |
| `TRAIN_RATIO` | 0.7 | 训练集比例 |
| `SEED` | 42 | 随机种子 |

## 模型架构

- **编码器**：多层深度编码 — 沿 −z 方向记录最多 8 个表面穿透深度 + 累计厚度 + 占据掩码
- **扩散模型**：U-Net 条件扩散（DDPM），以深度编码为条件
- **采样器**：支持 DDPM 和 DDIM（确定性加速）两种采样方式
- **缺失数据处理**：
  - 置信度加权 Loss — 有效区域全权重，缺失区域低权重
  - 随机掩码增强 — 训练时额外遮盖部分有效点
  - 引导式采样 — 已知区域向 GT 靠拢

## 常见问题

### Q: CUDA out of memory
减小 batch_size：
```bash
python train_diffusion.py --mode train --batch_size 2
```

### Q: 模型加载报错
确保 shapes 和 maps 目录结构一致，使用 `--skip_existing` 增量生成：
```bash
python shape2map.py --skip_existing
```

### Q: 生成的光路质量不理想
- 提高分辨率：`shape2map.py --resolution 256`
- 增加训练轮数：`train_diffusion.py --epochs 3000`
- 调整 KL 权重：`--kl_weight 0.005`

## 许可证

本项目仅用于学术研究目的。
