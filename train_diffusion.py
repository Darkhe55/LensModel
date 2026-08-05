"""
棱镜模型 - 条件扩散模型训练: Shape → Light Map
=================================================
从 3D 棱镜形状预测光路映射 (2D 位移场)。

Shape 编码: 多层深度编码 (Layered Depth Encoding)
  - 通道 0..K-1: 沿 -z 方向所有表面交点的归一化深度 (最多 K=8 层)
  - 通道 K: 累计材料厚度
  - 通道 K+1: 材料占据掩码
  完整捕获孔洞、多层穿越等复杂拓扑。

Map 目标: 归一化位移场 (H, W, 2) = output_grid - input_grid

缺失数据处理:
  - 置信度加权 loss (有效区域全权重, 缺失区域极小权重保持平滑)
  - 随机掩码增强 (训练时额外遮盖部分有效点, 教模型补全)
  - 引导式采样 (已知区域向 GT 靠拢, 缺失区域由模型推断)

用法:
  python train_diffusion.py                        # 完整流程
  python train_diffusion.py --mode train           # 仅训练
  python train_diffusion.py --mode sample          # 仅采样
  python train_diffusion.py --mode evaluate        # 仅评估
  python train_diffusion.py --epochs 2000 --batch_size 4

# 默认 KL 权重 1e-3
python train_diffusion.py --mode train

# 调大/调小 KL 权重
python train_diffusion.py --mode train --kl_weight 0.01
python train_diffusion.py --mode train --kl_weight 0.0   # 关闭 KL, 退回纯 MSE
"""
import os, sys, math, json, argparse, random, time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================== 配置 =====================
IMG_SIZE = 64               # 工作分辨率
CHANNELS_OUT = 2            # 输出通道 (dx, dy)
K_LAYERS = 8                # 最多记录 8 个表面交点
COND_CHANNELS = K_LAYERS + 2 + 2  # 深度层(8) + 厚度(1) + 掩码(1) + valid(1) + confidence(1) = 12
T_STEPS = 1000              # 扩散步数
BETA_START = 1e-4
BETA_END = 0.02
LEARNING_RATE = 2e-4
DEFAULT_EPOCHS = 1500
DEFAULT_BATCH = 8
HIDDEN_DIM = 64             # U-Net 基础通道
RANDOM_MASK_RATIO = 0.15    # 训练时随机额外遮盖比例
GUIDANCE_STRENGTH = 0.7     # 采样引导强度
SEED = 42

# ---- 参数追踪阈值 (由损失收敛判定阈值反推) ----
# 损失收敛判定: 相对变化低于 LOSS_CONVERGENCE_EPS 视为已收敛
# 由一阶近似 |ΔL| ≤ ||g||·||Δθ|| (梯度裁剪 ≤ 1.0), 该量级以下的参数漂移
# 对损失的影响低于收敛判定精度 → 单参数相对变化阈值取同一量级
LOSS_CONVERGENCE_EPS = 1e-3
PARAM_DELTA_RATIO = LOSS_CONVERGENCE_EPS  # 单参数相对变化阈值
MIN_TRIGGER_STEPS = 10      # 触发下限: 至少累积 10 步典型 Adam 位移 (lr·√n)

BASE_DIR = Path(__file__).parent
SHAPES_DIR = BASE_DIR / "shapes"
MAPS_DIR = BASE_DIR / "maps"
CHECKPOINT_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# ===================== 网格加载 =====================
# 复用 shape2map.py 中支持 ASCII / binary_little_endian / binary_big_endian 的健壮加载器
from shape2map import load_ply, load_obj, load_model


# ===================== 多层深度编码 =====================
def compute_layered_depth(vertices, faces, resolution, k_layers=K_LAYERS):
    """
    多层深度编码: 在每个 (x,y) 处沿 -z 发射光线，记录所有表面交点。
    返回 (resolution, resolution, k_layers+2):
      [:,:,0:K]  = 归一化交点深度 (从顶到底)
      [:,:,K]    = 累计材料厚度
      [:,:,K+1]  = 材料占据掩码
    """
    from scipy.ndimage import maximum_filter, binary_dilation

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    margin = 0.15 * max(vmax[0] - vmin[0], vmax[1] - vmin[1])
    x_min, x_max = vmin[0] - margin, vmax[0] + margin
    y_min, y_max = vmin[1] - margin, vmax[1] + margin
    z_min, z_max = vmin[2], vmax[2]
    z_span = max(z_max - z_min, 1e-8)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    # dir = (0,0,-1), h = cross(dir, edge2) = (edge2[:,1], -edge2[:,0], 0)
    h = np.stack([edge2[:, 1], -edge2[:, 0], np.zeros(len(edge2))], axis=1)
    a = np.sum(edge1 * h, axis=1)
    valid_tri = np.abs(a) > 1e-10
    f_inv = np.zeros_like(a)
    f_inv[valid_tri] = 1.0 / a[valid_tri]

    encoding = np.zeros((resolution, resolution, k_layers + 2), dtype=np.float32)
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)

    for i in range(resolution):
        origins = np.stack([xs, np.full(resolution, ys[i]),
                            np.full(resolution, z_max + 10.0)], axis=1)
        s = origins[:, np.newaxis, :] - v0[np.newaxis, :, :]
        u = f_inv[np.newaxis, :] * np.sum(s * h[np.newaxis, :, :], axis=2)
        q = np.cross(s, edge1[np.newaxis, :, :])
        v_param = f_inv[np.newaxis, :] * (-q[:, :, 2])
        t_param = f_inv[np.newaxis, :] * np.sum(edge2[np.newaxis, :, :] * q, axis=2)

        hit = (valid_tri[np.newaxis, :] &
               (u >= -1e-8) & (u <= 1.0 + 1e-8) &
               (v_param >= -1e-8) & (u + v_param <= 1.0 + 1e-8) &
               (t_param > 1e-8))

        z_vals = (z_max + 10.0) - t_param
        z_vals[~hit] = -np.inf

        for j in range(resolution):
            hits_z = z_vals[j][z_vals[j] > -np.inf]
            if len(hits_z) == 0:
                continue
            hits_z = np.sort(hits_z)[::-1]
            # 去重
            unique_z = [hits_z[0]]
            for z in hits_z[1:]:
                if abs(z - unique_z[-1]) > 1e-5:
                    unique_z.append(z)
            n_hits = min(len(unique_z), k_layers)
            for k in range(n_hits):
                encoding[i, j, k] = (unique_z[k] - z_min) / z_span
            # 累计厚度
            thickness = 0.0
            for k in range(0, n_hits - 1, 2):
                thickness += (unique_z[k] - unique_z[k + 1]) / z_span
            encoding[i, j, k_layers] = thickness
            encoding[i, j, k_layers + 1] = 1.0 if n_hits >= 2 else 0.0

    return encoding


# ===================== 数据收集与划分 =====================
def collect_pairs():
    """收集所有 (shape, map) 配对"""
    pairs = []
    if not MAPS_DIR.exists():
        return pairs
    for sub in sorted(MAPS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        shape_sub = SHAPES_DIR / sub.name
        if not shape_sub.exists():
            continue
        for npz_file in sorted(sub.glob("*_lightmap.npz")):
            stem = npz_file.stem.replace("_lightmap", "")
            shape_file = None
            for ext in ['.ply', '.obj']:
                candidate = shape_sub / f"{stem}{ext}"
                if candidate.exists():
                    shape_file = candidate
                    break
            if shape_file:
                pairs.append({
                    'shape_path': str(shape_file),
                    'map_path': str(npz_file),
                    'category': sub.name,
                    'name': stem,
                })
    return pairs


def split_data(pairs, train_ratio=0.7, val_ratio=0.15, seed=42):
    """分层抽样划分"""
    rng = random.Random(seed)
    by_cat = {}
    for p in pairs:
        by_cat.setdefault(p['category'], []).append(p)
    train, val, test = [], [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ===================== 数据集 =====================
def _resize_2d(arr, size):
    from scipy.ndimage import zoom
    h, w = arr.shape[:2]
    if h == size and w == size:
        return arr
    factor = size / h
    if arr.ndim == 3:
        return zoom(arr, (factor, factor, 1), order=1)
    return zoom(arr, factor, order=1)


class PrismDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.cache = {}

    def __len__(self):
        return len(self.pairs)

    def _load_sample(self, idx):
        info = self.pairs[idx]
        key = info['map_path']
        if key in self.cache:
            return self.cache[key]

        # 加载 map
        data = np.load(info['map_path'])
        output_grid = data['output_grid']
        input_grid = data['input_grid']
        valid_mask = data['valid_mask'].astype(np.float32)

        displacement = output_grid - input_grid
        displacement[~data['valid_mask']] = 0.0
        disp_max = np.abs(displacement[data['valid_mask']]).max() if data['valid_mask'].sum() > 0 else 1.0
        disp_max = max(float(disp_max), 1e-6)
        displacement_norm = displacement / disp_max

        # 加载 shape → 多层深度编码
        vertices, faces = load_model(info['shape_path'])
        if vertices is None or len(faces) == 0:
            encoding = np.zeros((self.img_size, self.img_size, K_LAYERS + 2), dtype=np.float32)
        else:
            vertices[:, 2] -= vertices[:, 2].min()
            encoding = compute_layered_depth(vertices, faces, self.img_size, K_LAYERS)

        sample = {
            'encoding': encoding.astype(np.float32),
            'displacement': displacement_norm.astype(np.float32),
            'valid_mask': valid_mask,
            'disp_scale': np.float32(disp_max),
        }
        self.cache[key] = sample
        return sample

    def __getitem__(self, idx):
        sample = self._load_sample(idx)
        sample_name = self.pairs[idx]['name']
        enc = sample['encoding'].copy()
        disp = sample['displacement'].copy()
        mask = sample['valid_mask'].copy()

        # 缩放 (enc 已在 _load_sample 按 img_size 计算; disp/mask 来自原始分辨率, 需各自独立缩放)
        if enc.shape[0] != self.img_size:
            enc = _resize_2d(enc, self.img_size)
        if disp.shape[0] != self.img_size:
            disp = _resize_2d(disp, self.img_size)
        if mask.shape[0] != self.img_size:
            mask = _resize_2d(mask, self.img_size)

        # 几何增强
        if self.augment:
            k = random.randint(0, 3)
            if k > 0:
                enc = np.rot90(enc, k, axes=(0, 1)).copy()
                disp = np.rot90(disp, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k).copy()
                for _ in range(k):
                    disp_new = np.zeros_like(disp)
                    disp_new[:, :, 0] = -disp[:, :, 1]
                    disp_new[:, :, 1] = disp[:, :, 0]
                    disp = disp_new
            if random.random() > 0.5:
                enc = np.fliplr(enc).copy()
                disp = np.fliplr(disp).copy()
                disp[:, :, 0] *= -1
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                enc = np.flipud(enc).copy()
                disp = np.flipud(disp).copy()
                disp[:, :, 1] *= -1
                mask = np.flipud(mask).copy()

        # 随机掩码增强: 额外遮盖部分有效点
        if self.augment and RANDOM_MASK_RATIO > 0:
            valid_idx = np.argwhere(mask > 0.5)
            n_drop = int(len(valid_idx) * RANDOM_MASK_RATIO)
            if n_drop > 0 and len(valid_idx) > 0:
                drop = np.random.choice(len(valid_idx), n_drop, replace=False)
                for di in drop:
                    r, c = valid_idx[di]
                    mask[r, c] = 0.0
                    disp[r, c, :] = 0.0

        # 置信度图 (mask 的平滑版)
        from scipy.ndimage import uniform_filter
        confidence = uniform_filter(mask, size=3)
        confidence = np.clip(confidence * 1.5, 0, 1).astype(np.float32)

        # 组装条件: encoding(K+2) + mask(1) + confidence(1)
        cond = np.concatenate([enc, mask[:, :, None], confidence[:, :, None]], axis=-1)

        cond_t = torch.from_numpy(cond).permute(2, 0, 1).float()
        target_t = torch.from_numpy(disp).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        conf_t = torch.from_numpy(confidence).unsqueeze(0).float()
        # 第 5 个返回值: 样本名 (用于参数追踪日志)
        return cond_t, target_t, mask_t, conf_t, sample_name


# ===================== 扩散调度 =====================
class DiffusionSchedule:
    def __init__(self, t_steps=T_STEPS, beta_start=BETA_START, beta_end=BETA_END):
        self.T = t_steps
        self.betas = torch.linspace(beta_start, beta_end, t_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.posterior_var = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        self.device = torch.device('cpu')

    def to(self, device):
        """将调度表所有张量移到指定设备 (GPU 上直接索引计算, 避免 CPU/GPU 来回搬运)"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus = self.sqrt_one_minus.to(device)
        self.sqrt_recip_alpha = self.sqrt_recip_alpha.to(device)
        self.posterior_var = self.posterior_var.to(device)
        return self

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        sqrt_1mac = self.sqrt_one_minus[t].view(-1, 1, 1, 1).to(x0.device)
        return sqrt_ac * x0 + sqrt_1mac * noise, noise

    def _extract(self, a, t, x_shape):
        """按时间步 t 从调度表 a 取值, reshape 为可广播形状 (B,1,1,...)"""
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_posterior_mean(self, x0, xt, t):
        """真实后验 q(x_{t-1}|x_t,x_0) 的均值 μ̃_t 及其系数"""
        ac = self._extract(self.alpha_cumprod, t, x0.shape)
        ac_prev = self._extract(self.alpha_cumprod_prev, t, x0.shape)
        alpha = self._extract(self.alphas, t, x0.shape)
        beta = self._extract(self.betas, t, x0.shape)
        coef1 = beta * torch.sqrt(ac_prev) / (1.0 - ac)
        coef2 = (1.0 - ac_prev) * torch.sqrt(alpha) / (1.0 - ac)
        return coef1 * x0 + coef2 * xt, coef1, coef2, ac

    def kl_loss(self, x0, xt, t, pred_noise):
        """
        变分下界 KL 辅助损失 L_vb:
          KL( q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t) )
        反向方差固定为后验方差 β̃_t 时, KL 退化为两高斯均值差的平方项。
        x0: 干净位移场(target)  xt: 加噪后  t: 时间步  pred_noise: 预测噪声 ε_θ
        """
        # 真实后验均值 (用真实 x0)
        true_mean, coef1, coef2, ac = self.q_posterior_mean(x0, xt, t)
        # 由预测噪声反推 x0, 再算学习的反向均值 μ_θ
        x0_pred = (xt - torch.sqrt(1.0 - ac) * pred_noise) / torch.sqrt(ac)
        pred_mean = coef1 * x0_pred + coef2 * xt
        # 后验方差 β̃_t (clamp 避免 t=0 除零)
        posterior_var = self._extract(self.posterior_var, t, x0.shape).clamp(min=1e-8)
        kl = (true_mean - pred_mean) ** 2 / (2.0 * posterior_var)
        return kl.mean()

    @torch.no_grad()
    def p_sample(self, model, xt, t, cond):
        t_tensor = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        pred_noise = model(xt, t_tensor, cond)
        beta_t = self.betas[t].to(xt.device)
        sqrt_1mac = self.sqrt_one_minus[t].to(xt.device)
        sqrt_recip = self.sqrt_recip_alpha[t].to(xt.device)
        mean = sqrt_recip * (xt - beta_t / sqrt_1mac * pred_noise)
        if t == 0:
            return mean
        var = self.posterior_var[t].to(xt.device)
        return mean + torch.sqrt(var) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, model, shape, cond, device='cpu'):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond)
        return x

    @torch.no_grad()
    def sample_guided(self, model, shape, cond, known_target, known_mask,
                      strength=GUIDANCE_STRENGTH, device='cpu'):
        """引导式采样: 已知区域向GT靠拢, 缺失区域由模型推断"""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond)
            if t > 0 and known_target is not None:
                alpha_t = self.alpha_cumprod[t].to(device)
                s = strength * (1.0 - alpha_t)
                x = known_mask * (s * known_target + (1 - s) * x) + (1 - known_mask) * x
        return x


# ===================== U-Net 模型 =====================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

def _num_groups(channels, target=8):
    """找到不超过 target 且能整除 channels 的最大组数 (GroupNorm 要求通道数可被组数整除)"""
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(_num_groups(in_ch), in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(_num_groups(out_ch), out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(ch), ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class CondUNet(nn.Module):
    """条件 U-Net: 接受多通道条件输入, 预测噪声"""
    def __init__(self, in_ch=2, out_ch=2, cond_ch=COND_CHANNELS, base_dim=HIDDEN_DIM):
        super().__init__()
        time_dim = base_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        # 条件编码器 (多层深度 + mask + confidence)
        self.cond_enc = nn.Sequential(
            nn.Conv2d(cond_ch, base_dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1), nn.SiLU(),
        )
        # Encoder
        self.enc1 = ResBlock(base_dim + in_ch, base_dim, time_dim)
        self.enc2 = ResBlock(base_dim, base_dim * 2, time_dim)
        self.enc3 = ResBlock(base_dim * 2, base_dim * 4, time_dim)
        self.attn = AttentionBlock(base_dim * 4)
        self.down1 = nn.Conv2d(base_dim, base_dim, 4, 2, 1)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 2, 4, 2, 1)
        # Bottleneck
        self.bot = ResBlock(base_dim * 4, base_dim * 4, time_dim)
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, 2, 1)
        self.dec2 = ResBlock(base_dim * 4, base_dim * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, 2, 1)
        self.dec1 = ResBlock(base_dim * 2, base_dim, time_dim)
        self.final = nn.Conv2d(base_dim, out_ch, 1)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t)
        c = self.cond_enc(cond)
        x = torch.cat([x, c], dim=1)
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h3 = self.enc3(self.down2(h2), t_emb)
        h3 = self.attn(h3)
        b = self.bot(h3, t_emb)
        u2 = self.up2(b)
        if u2.shape[2:] != h2.shape[2:]:
            u2 = F.interpolate(u2, size=h2.shape[2:])
        d2 = self.dec2(torch.cat([u2, h2], dim=1), t_emb)
        u1 = self.up1(d2)
        if u1.shape[2:] != h1.shape[2:]:
            u1 = F.interpolate(u1, size=h1.shape[2:])
        d1 = self.dec1(torch.cat([u1, h1], dim=1), t_emb)
        return self.final(d1)


# ===================== 训练 =====================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    pairs = collect_pairs()
    print(f"找到 {len(pairs)} 对 shape↔map 样本")
    if len(pairs) < 4:
        print("错误: 样本太少，请先运行 shape2map.py 生成数据")
        return

    train_pairs, val_pairs, test_pairs = split_data(pairs, seed=SEED)
    print(f"划分: 训练 {len(train_pairs)} / 验证 {len(val_pairs)} / 测试 {len(test_pairs)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "data_split.json", 'w', encoding='utf-8') as f:
        json.dump({
            'train': [p['name'] for p in train_pairs],
            'val': [p['name'] for p in val_pairs],
            'test': [p['name'] for p in test_pairs],
        }, f, ensure_ascii=False, indent=2)

    train_ds = PrismDataset(train_pairs, IMG_SIZE, augment=True)
    val_ds = PrismDataset(val_pairs, IMG_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CondUNet(in_ch=CHANNELS_OUT, out_ch=CHANNELS_OUT,
                     cond_ch=COND_CHANNELS, base_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    diffusion = DiffusionSchedule().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print(f"条件通道: {COND_CHANNELS} (深度{K_LAYERS} + 厚度1 + 掩码1 + valid1 + conf1)")

    # ---- 参数追踪 (差值存储): 每参数基线 = 其上次触发保存节点, 累积达阈值才存差值 ----
    delta_ratio = getattr(args, 'delta_ratio', PARAM_DELTA_RATIO)
    min_trigger_steps = max(1, getattr(args, 'min_trigger_steps', MIN_TRIGGER_STEPS))
    anchor_every = max(0, getattr(args, 'anchor_every', 0))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    track_dir = CHECKPOINT_DIR / "param_tracking" / run_id
    track_params_dir = track_dir / "params"
    track_params_dir.mkdir(parents=True, exist_ok=True)
    track_log_path = track_dir / "update_log.jsonl"       # 每步训练详情
    delta_log_path = track_dir / "delta_log.jsonl"        # 每次触发保存的差值事件
    last_trigger_path = track_dir / "param_last_trigger.json"  # 每参数最后一次达阈值的时间点
    track_method = {
        'model': 'CondUNet (多层深度编码条件 U-Net, 噪声预测)',
        'loss': '置信度加权 MSE (主损失) + KL 散度辅助损失 L_vb (变分下界)',
        'optimizer': f'AdamW (lr={LEARNING_RATE}, weight_decay=1e-4)',
        'lr_scheduler': f'CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)',
        'diffusion': f'DDPM q_sample 前向加噪, T={T_STEPS}, beta=[{BETA_START},{BETA_END}]',
        'grad_clip': 'clip_grad_norm_ max_norm=1.0',
        'augmentation': f'rot90/flip + 随机掩码 (ratio={RANDOM_MASK_RATIO})',
        'param_storage': '差值存储: 每参数以其上次触发保存节点为基线, 累积变化达阈值才保存差值',
        'loss_convergence_eps': LOSS_CONVERGENCE_EPS,
        'param_delta_ratio': delta_ratio,
        'min_trigger_steps': min_trigger_steps,
        'anchor_every': anchor_every,
        'reconstruction': 'θ(t) = 最近 anchor + 按 step 顺序累加 delta; 训练结束 final_flush 存剩余累积并写末锚点',
        'kl_weight': args.kl_weight,
        'batch_size': args.batch_size,
        'seed': SEED,
        'num_params': n_params,
        'train_samples': [p['name'] for p in train_pairs],
        'val_samples': [p['name'] for p in val_pairs],
    }
    with open(track_dir / "run_config.json", 'w', encoding='utf-8') as f:
        json.dump({'run_id': run_id, 'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                   **track_method}, f, ensure_ascii=False, indent=2)

    # 单参数阈值 = max(相对阈值×基线范数, 下限步数×典型Adam位移)
    def _param_threshold(base_norm, numel):
        rel = delta_ratio * max(base_norm, 1e-8)
        floor = min_trigger_steps * LEARNING_RATE * math.sqrt(numel)
        return max(rel, floor)

    # 初始基线 (每个参数的起点 = 当前参数, 视为 step 0 锚点触发)
    baseline = {n: p.detach().clone() for n, p in model.named_parameters()}
    baseline_norm = {n: torch.linalg.vector_norm(b).item() for n, b in baseline.items()}
    thresholds = {n: _param_threshold(baseline_norm[n], p.numel())
                  for n, p in model.named_parameters()}
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    last_trigger = {n: {'global_step': 0, 'timestamp': now_str, 'epoch': 0,
                        'type': 'anchor', 'cumulative_norm': 0.0,
                        'threshold': thresholds[n]} for n in baseline}
    with open(last_trigger_path, 'w', encoding='utf-8') as f:
        json.dump(last_trigger, f, ensure_ascii=False, indent=2)

    def _save_anchor(step, epoch):
        torch.save({'global_step': step, 'epoch': epoch, 'type': 'anchor',
                    'model_state_dict': model.state_dict()},
                   track_params_dir / f"anchor_step_{step:08d}.pt")

    # 初始全量锚点 (差值重建的起点)
    _save_anchor(0, 0)
    last_anchor_step = 0

    print(f"参数追踪 (差值存储): 相对阈值={delta_ratio} (由损失收敛容差 {LOSS_CONVERGENCE_EPS} 反推) → {track_dir}")
    print(f"开始训练 {args.epochs} epochs ...\n")

    best_val_loss = float('inf')
    global_step = 0          # 反向传播总次数
    saved_step = 0           # 已保存的差值文件数
    name2cat = {p['name']: p['category'] for p in train_pairs}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_kl = 0.0
        for batch_idx, (cond, target, mask, confidence, names) in enumerate(train_loader):
            cond = cond.to(device)
            target = target.to(device)
            mask = mask.to(device)
            confidence = confidence.to(device)

            t = torch.randint(0, diffusion.T, (target.shape[0],), device=device)
            noise = torch.randn_like(target)
            xt, _ = diffusion.q_sample(target, t, noise)
            pred = model(xt, t, cond)

            # 置信度加权 MSE loss (主损失)
            weight = confidence * mask + 0.05 * (1 - mask)
            weight = weight.expand_as(pred)
            loss_mse = (weight * (pred - noise) ** 2).sum() / (weight.sum() + 1e-8)

            # KL 散度辅助损失 (变分下界 L_vb)
            loss_kl = diffusion.kl_loss(target, xt, t, pred)

            # 加权组合
            loss = loss_mse + args.kl_weight * loss_kl

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            train_loss += loss.item()
            train_kl += loss_kl.item()

            # ---- 记录每步训练详情 (样本/损失/时间步等) ----
            with open(track_log_path, 'a', encoding='utf-8') as lf:
                lf.write(json.dumps({
                    'global_step': global_step,
                    'epoch': epoch,
                    'batch_in_epoch': batch_idx,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'elapsed_sec': round(time.time() - t_start, 2),
                    'samples': [{'name': sn, 'category': name2cat.get(sn, '?')} for sn in names],
                    'sample_names': list(names),
                    'diffusion_timesteps': [int(v) for v in t.detach().cpu().tolist()],
                    'loss_total': float(loss.item()),
                    'loss_mse': float(loss_mse.item()),
                    'loss_kl': float(loss_kl.item()),
                    'kl_weight': args.kl_weight,
                    'grad_norm_before_clip': float(grad_norm.item()),
                    'lr': float(scheduler.get_last_lr()[0]),
                    'optimizer': 'AdamW',
                    'method': '置信度加权MSE + KL(L_vb), DDPM噪声预测, 梯度裁剪1.0',
                    'augment': True,
                    'device': str(device),
                }, ensure_ascii=False) + '\n')

            # ---- 差值追踪: 基线 = 该参数上次触发保存节点, 累积达阈值才存 ----
            triggered_deltas, triggered_norms = {}, {}
            ts_now = time.strftime("%Y-%m-%d %H:%M:%S")
            for pname, p in model.named_parameters():
                diff = p.data - baseline[pname]
                c_norm = torch.linalg.vector_norm(diff).item()
                if c_norm >= thresholds[pname]:
                    triggered_deltas[pname] = diff.detach().cpu().clone()
                    triggered_norms[pname] = c_norm
                    baseline[pname] = p.data.detach().clone()
                    baseline_norm[pname] = torch.linalg.vector_norm(baseline[pname]).item()
                    thresholds[pname] = _param_threshold(baseline_norm[pname], p.numel())
                    last_trigger[pname] = {
                        'global_step': global_step, 'timestamp': ts_now,
                        'epoch': epoch, 'type': 'delta',
                        'cumulative_norm': c_norm,
                        'threshold': thresholds[pname],
                    }
            if triggered_deltas:
                saved_step += 1
                torch.save({'global_step': global_step, 'epoch': epoch,
                            'type': 'delta', 'deltas': triggered_deltas,
                            'norms': triggered_norms},
                           track_params_dir / f"delta_step_{global_step:08d}.pt")
                with open(last_trigger_path, 'w', encoding='utf-8') as f:
                    json.dump(last_trigger, f, ensure_ascii=False, indent=2)
                with open(delta_log_path, 'a', encoding='utf-8') as lf:
                    lf.write(json.dumps({
                        'global_step': global_step, 'epoch': epoch, 'timestamp': ts_now,
                        'num_triggered': len(triggered_deltas),
                        'params': list(triggered_deltas.keys()),
                        'norms': triggered_norms,
                        'sample_names': list(names),
                        'loss_total': float(loss.item()),
                        'param_file': f"params/delta_step_{global_step:08d}.pt",
                    }, ensure_ascii=False) + '\n')

            # 周期性全量锚点 (可选, 防止差值链过长)
            if anchor_every > 0 and global_step - last_anchor_step >= anchor_every:
                _save_anchor(global_step, epoch)
                last_anchor_step = global_step

        scheduler.step()
        train_loss /= len(train_loader)
        train_kl /= len(train_loader)

        # 验证
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for cond, target, mask, confidence, _names in val_loader:
                    cond, target = cond.to(device), target.to(device)
                    mask, confidence = mask.to(device), confidence.to(device)
                    t = torch.randint(0, diffusion.T, (target.shape[0],), device=device)
                    noise = torch.randn_like(target)
                    xt, _ = diffusion.q_sample(target, t, noise)
                    pred = model(xt, t, cond)
                    weight = confidence * mask + 0.05 * (1 - mask)
                    weight = weight.expand_as(pred)
                    val_loss += ((weight * (pred - noise) ** 2).sum() / (weight.sum() + 1e-8)).item()
            val_loss /= max(len(val_loader), 1)

            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{args.epochs} | train={train_loss:.5f} | kl={train_kl:.5f} | "
                  f"val={val_loss:.5f} | lr={lr_now:.2e} | {elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': {
                        'img_size': IMG_SIZE, 'k_layers': K_LAYERS,
                        'cond_channels': COND_CHANNELS, 'hidden_dim': HIDDEN_DIM,
                        't_steps': T_STEPS,
                    }
                }, CHECKPOINT_DIR / "best_model.pt")

    # ---- 训练结束: 落盘未达阈值的剩余累积, 保证最终参数可精确重建 ----
    final_deltas = {}
    for pname, p in model.named_parameters():
        d = p.data - baseline[pname]
        if torch.linalg.vector_norm(d).item() > 0.0:
            final_deltas[pname] = d.detach().cpu().clone()
    if final_deltas:
        ts_now = time.strftime("%Y-%m-%d %H:%M:%S")
        torch.save({'global_step': global_step, 'epoch': args.epochs,
                    'type': 'final_flush', 'deltas': final_deltas},
                   track_params_dir / f"delta_step_{global_step:08d}_final.pt")
        for pname in final_deltas:
            last_trigger[pname] = {
                'global_step': global_step, 'timestamp': ts_now,
                'epoch': args.epochs, 'type': 'final_flush',
                'cumulative_norm': float(torch.linalg.vector_norm(final_deltas[pname]).item()),
                'threshold': thresholds[pname],
            }
        with open(last_trigger_path, 'w', encoding='utf-8') as f:
            json.dump(last_trigger, f, ensure_ascii=False, indent=2)
    _save_anchor(global_step, args.epochs)

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
    }, CHECKPOINT_DIR / "final_model.pt")
    print(f"\n训练完成! 最佳 val_loss={best_val_loss:.5f}, 耗时 {time.time()-t_start:.0f}s")
    print(f"模型: {CHECKPOINT_DIR}")
    print(f"参数追踪: 共 {saved_step} 份差值文件 + 锚点 + 日志 → {track_dir}")


# ===================== 采样 =====================
@torch.no_grad()
def sample(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondUNet(in_ch=CHANNELS_OUT, out_ch=CHANNELS_OUT,
                     cond_ch=COND_CHANNELS, base_dim=HIDDEN_DIM).to(device)

    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "final_model.pt"
    if not ckpt_path.exists():
        print("错误: 无模型文件，请先训练")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"加载模型: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    diffusion = DiffusionSchedule().to(device)
    pairs = collect_pairs()
    _, _, test_pairs = split_data(pairs, seed=SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"在测试集 ({len(test_pairs)} 个) 上生成预测...")

    for info in test_pairs:
        vertices, faces = load_model(info['shape_path'])
        if vertices is None or len(faces) == 0:
            continue
        vertices[:, 2] -= vertices[:, 2].min()
        enc = compute_layered_depth(vertices, faces, IMG_SIZE, K_LAYERS)

        # 加载 GT 用于引导
        data = np.load(info['map_path'])
        gt_disp = data['output_grid'] - data['input_grid']
        valid_mask = data['valid_mask'].astype(np.float32)
        disp_max = np.abs(gt_disp[data['valid_mask']]).max() if data['valid_mask'].sum() > 0 else 1.0
        disp_max = max(float(disp_max), 1e-6)
        gt_norm = gt_disp / disp_max

        # 缩放到工作分辨率
        mask_r = _resize_2d(valid_mask, IMG_SIZE)
        gt_r = _resize_2d(gt_norm, IMG_SIZE)
        from scipy.ndimage import uniform_filter
        conf_r = np.clip(uniform_filter(mask_r, size=3) * 1.5, 0, 1).astype(np.float32)

        # 条件
        cond_np = np.concatenate([enc, mask_r[:, :, None], conf_r[:, :, None]], axis=-1)
        cond = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        known = torch.from_numpy(gt_r).permute(2, 0, 1).unsqueeze(0).float().to(device)
        known_m = torch.from_numpy(mask_r).unsqueeze(0).unsqueeze(0).float().to(device)

        # 引导采样
        pred = diffusion.sample_guided(model, (1, CHANNELS_OUT, IMG_SIZE, IMG_SIZE),
                                       cond, known, known_m, GUIDANCE_STRENGTH, device)
        pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)

        np.savez_compressed(
            RESULTS_DIR / f"{info['name']}_prediction.npz",
            predicted_displacement=pred_np * disp_max,
            predicted_normalized=pred_np,
            gt_displacement=gt_disp,
            encoding=enc,
            valid_mask=valid_mask,
            disp_scale=disp_max,
        )
        print(f"  [OK] {info['name']}")

    print(f"结果: {RESULTS_DIR}")


# ===================== 评估 =====================
@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondUNet(in_ch=CHANNELS_OUT, out_ch=CHANNELS_OUT,
                     cond_ch=COND_CHANNELS, base_dim=HIDDEN_DIM).to(device)

    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        print("错误: 无模型文件")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = DiffusionSchedule().to(device)
    pairs = collect_pairs()
    _, _, test_pairs = split_data(pairs, seed=SEED)
    test_ds = PrismDataset(test_pairs, IMG_SIZE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    mse_valid, mae_valid, n_valid = 0.0, 0.0, 0
    mse_missing, n_missing = 0.0, 0

    print(f"评估测试集 ({len(test_pairs)} 个样本)...")
    for cond, target, mask, confidence, _names in test_loader:
        cond, target = cond.to(device), target.to(device)
        mask = mask.to(device)

        # 引导采样
        pred = diffusion.sample_guided(model, target.shape, cond,
                                       target, mask, GUIDANCE_STRENGTH, device)
        diff = (pred - target)
        diff_sq = diff ** 2

        # 有效区域
        valid_region = (mask > 0.5).expand_as(diff)
        n_v = valid_region.sum().item()
        if n_v > 0:
            mse_valid += diff_sq[valid_region].sum().item()
            mae_valid += diff.abs()[valid_region].sum().item()
            n_valid += int(n_v)

        # 缺失区域
        missing_region = (mask < 0.5).expand_as(diff)
        n_m = missing_region.sum().item()
        if n_m > 0:
            mse_missing += diff_sq[missing_region].sum().item()
            n_missing += int(n_m)

    print(f"\n{'='*50}")
    print(f"  有效区域 ({n_valid} 像素):")
    print(f"    MSE:  {mse_valid / max(n_valid, 1):.6f}")
    print(f"    MAE:  {mae_valid / max(n_valid, 1):.6f}")
    print(f"    RMSE: {math.sqrt(mse_valid / max(n_valid, 1)):.6f}")
    if n_missing > 0:
        print(f"  缺失区域 ({n_missing} 像素):")
        print(f"    MSE:  {mse_missing / n_missing:.6f} (补全参考)")
    print(f"{'='*50}")


# ===================== 入口 =====================
def main():
    parser = argparse.ArgumentParser(description="Shape→Map 条件扩散模型 (多层深度编码)")
    parser.add_argument("--mode", choices=['train', 'sample', 'evaluate', 'all'], default='all')
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--kl_weight", type=float, default=1e-3,
                        help="KL 散度辅助损失权重 (变分下界 L_vb, 默认 1e-3)")
    parser.add_argument("--delta_ratio", type=float, default=PARAM_DELTA_RATIO,
                        help="单参数相对变化阈值 (默认 1e-3, 由损失收敛容差反推)")
    parser.add_argument("--min_trigger_steps", type=int, default=MIN_TRIGGER_STEPS,
                        help="触发下限步数: 阈值不低于该步数×典型Adam位移 (默认 10)")
    parser.add_argument("--anchor_every", type=int, default=0,
                        help="每 N 步额外存一次全量锚点并防差值链过长 (默认 0=仅初始/结束锚点)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print(f"{'='*50}")
    print(f"  棱镜光路映射 - 条件扩散模型")
    print(f"  分辨率: {IMG_SIZE}x{IMG_SIZE} | 扩散步数: {T_STEPS}")
    print(f"  深度层数: {K_LAYERS} | 条件通道: {COND_CHANNELS}")
    print(f"{'='*50}\n")

    if args.mode in ('train', 'all'):
        train(args)
    if args.mode in ('sample', 'all'):
        sample(args)
    if args.mode in ('evaluate', 'all'):
        evaluate(args)


if __name__ == "__main__":
    main()
