"""
棱镜模型 - 单形状推理脚本
==========================
对任意给定的 3D 模型文件，预测其光路映射。
支持无 GT 的纯生成模式 (不需要对应的 map 文件)。

运行:
  python inference_diffusion.py shapes/01_正多面体/正四面体.ply
  python inference_diffusion.py shapes/05_动物/兔子.ply --output results/兔子_pred.npz
  python inference_diffusion.py shapes/04_多亏格曲面/环面_亏格1.ply --ddim_steps 50
"""
import os, sys, math, argparse, time
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from train_diffusion import (
    CondUNet, DiffusionSchedule,
    compute_layered_depth, load_model, _resize_2d
)


# ===================== DDIM 加速采样 =====================
class DDIMSampler:
    """DDIM 确定性采样 (比 DDPM 快 10-20x)"""
    def __init__(self, t_steps=T_STEPS, beta_start=BETA_START, beta_end=BETA_END):
        self.T = t_steps
        betas = torch.linspace(beta_start, beta_end, t_steps)
        alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)

    @torch.no_grad()
    def sample(self, model, shape, cond, ddim_steps=50, eta=0.0, device='cpu'):
        """
        DDIM 采样
        ddim_steps: 实际去噪步数 (远小于 T)
        eta: 随机性 (0=确定性, 1=DDPM)
        """
        # 均匀选取时间步子集
        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        x = torch.randn(shape, device=device)
        alpha_cumprod = self.alpha_cumprod.to(device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor, cond)

            alpha_t = alpha_cumprod[t]
            # 预测 x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3, 3)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = alpha_cumprod[t_next]
                # DDIM 更新
                sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t)) * \
                        torch.sqrt(1 - alpha_t / alpha_next)
                dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * pred_noise
                x = torch.sqrt(alpha_next) * x0_pred + dir_xt
                if sigma > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x


# ===================== 推理 =====================
@torch.no_grad()
def infer(shape_path, output_path=None, ddim_steps=50, use_ddim=True):
    """
    对单个形状文件进行光路映射推理。
    shape_path: PLY/OBJ 文件路径
    output_path: 输出 NPZ 路径 (默认保存到 results/)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape_path = Path(shape_path)

    if not shape_path.exists():
        print(f"错误: 文件不存在 {shape_path}")
        return None

    # 加载模型
    model = CondUNet(in_ch=CHANNELS_OUT, out_ch=CHANNELS_OUT,
                     cond_ch=COND_CHANNELS, base_dim=HIDDEN_DIM).to(device)
    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "final_model.pt"
    if not ckpt_path.exists():
        print("错误: 无训练好的模型，请先运行 train_diffusion.py")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"模型: {ckpt_path.name} (epoch {ckpt.get('epoch', '?')})")

    # 加载形状
    vertices, faces = load_model(shape_path)
    if vertices is None or len(faces) == 0:
        print(f"错误: 无法加载模型 {shape_path}")
        return None
    vertices[:, 2] -= vertices[:, 2].min()
    print(f"形状: {shape_path.name} ({len(vertices)} 顶点, {len(faces)} 面)")

    # 多层深度编码
    t0 = time.time()
    enc = compute_layered_depth(vertices, faces, IMG_SIZE, K_LAYERS)
    encode_time = time.time() - t0
    print(f"编码耗时: {encode_time:.2f}s")

    # 无 GT 时: mask=全1 (假设全部需要预测), confidence=1
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    confidence = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    cond_np = np.concatenate([enc, mask[:, :, None], confidence[:, :, None]], axis=-1)
    cond = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 采样
    t0 = time.time()
    if use_ddim:
        sampler = DDIMSampler()
        pred = sampler.sample(model, (1, CHANNELS_OUT, IMG_SIZE, IMG_SIZE),
                              cond, ddim_steps=ddim_steps, device=device)
        print(f"DDIM 采样 ({ddim_steps} 步)")
    else:
        diffusion = DiffusionSchedule().to(device)
        pred = diffusion.sample(model, (1, CHANNELS_OUT, IMG_SIZE, IMG_SIZE), cond, device)
        print(f"DDPM 采样 ({T_STEPS} 步)")
    sample_time = time.time() - t0
    print(f"采样耗时: {sample_time:.2f}s")

    pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)

    # 保存
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f"{shape_path.stem}_inferred.npz"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        displacement=pred_np,
        encoding=enc,
        shape_path=str(shape_path),
        img_size=IMG_SIZE,
        ddim_steps=ddim_steps if use_ddim else T_STEPS,
        encode_time=encode_time,
        sample_time=sample_time,
    )
    print(f"输出: {output_path}")
    print(f"位移场范围: dx=[{pred_np[:,:,0].min():.3f}, {pred_np[:,:,0].max():.3f}], "
          f"dy=[{pred_np[:,:,1].min():.3f}, {pred_np[:,:,1].max():.3f}]")
    return pred_np


def main():
    parser = argparse.ArgumentParser(description="棱镜光路映射 - 单形状推理")
    parser.add_argument("shape", type=str, help="输入形状文件路径 (.ply 或 .obj)")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 NPZ 路径")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM 采样步数")
    parser.add_argument("--no_ddim", action="store_true", help="使用完整 DDPM (慢)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    infer(args.shape, args.output, args.ddim_steps, use_ddim=not args.no_ddim)


if __name__ == "__main__":
    main()
