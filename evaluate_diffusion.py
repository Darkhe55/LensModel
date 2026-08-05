"""
棱镜模型 - 测试与评估脚本
==========================
对训练好的扩散模型进行全面评估:
  - 逐样本精度指标 (MSE, MAE, RMSE, SSIM)
  - 有效区域 vs 缺失区域分别统计
  - 位移场方向/幅度误差
  - 生成结果平滑性检测
  - 批量推理耗时统计
  - 输出评估报告 JSON

运行:
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe evaluate_diffusion.py
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe evaluate.py --split val
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe evaluate.py --no_guidance
"""
import os, sys, math, json, argparse, time
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

# 导入共享模块
sys.path.insert(0, str(Path(__file__).parent))
from config import *
from train_diffusion import (
    CondUNet, DiffusionSchedule, PrismDataset,
    collect_pairs, split_data, compute_layered_depth,
    load_model, _resize_2d
)


# ===================== 指标计算 =====================
def compute_ssim_2d(pred, target, window_size=7):
    """简化 SSIM (结构相似性)"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    from scipy.ndimage import uniform_filter
    mu_p = uniform_filter(pred, size=window_size)
    mu_t = uniform_filter(target, size=window_size)
    sigma_p2 = uniform_filter(pred ** 2, size=window_size) - mu_p ** 2
    sigma_t2 = uniform_filter(target ** 2, size=window_size) - mu_t ** 2
    sigma_pt = uniform_filter(pred * target, size=window_size) - mu_p * mu_t
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p2 + sigma_t2 + C2))
    return float(ssim_map.mean())


def compute_metrics(pred, target, mask):
    """
    计算全面指标。
    pred, target: (H, W, 2) 位移场
    mask: (H, W) 有效区域
    """
    metrics = {}
    valid = mask > 0.5
    missing = ~valid

    diff = pred - target
    diff_mag = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)

    # 有效区域指标
    if valid.sum() > 0:
        d_valid = diff[valid]
        metrics['mse_valid'] = float((d_valid ** 2).mean())
        metrics['mae_valid'] = float(np.abs(d_valid).mean())
        metrics['rmse_valid'] = float(math.sqrt(metrics['mse_valid']))
        metrics['max_error_valid'] = float(np.abs(d_valid).max())
        metrics['magnitude_error_valid'] = float(diff_mag[valid].mean())
        # SSIM (逐通道)
        ssim_x = compute_ssim_2d(pred[:, :, 0] * valid, target[:, :, 0] * valid)
        ssim_y = compute_ssim_2d(pred[:, :, 1] * valid, target[:, :, 1] * valid)
        metrics['ssim_valid'] = (ssim_x + ssim_y) / 2
        metrics['n_valid_pixels'] = int(valid.sum())

    # 缺失区域 (补全质量参考)
    if missing.sum() > 0:
        d_missing = diff[missing]
        metrics['mse_missing'] = float((d_missing ** 2).mean())
        metrics['mae_missing'] = float(np.abs(d_missing).mean())
        metrics['n_missing_pixels'] = int(missing.sum())

    # 全局平滑性 (梯度范数)
    grad_x = np.diff(pred, axis=1)
    grad_y = np.diff(pred, axis=0)
    metrics['smoothness_pred'] = float(np.sqrt(grad_x[:-1, :, :] ** 2 + grad_y[:, :-1, :] ** 2).mean())
    grad_x_gt = np.diff(target, axis=1)
    grad_y_gt = np.diff(target, axis=0)
    metrics['smoothness_gt'] = float(np.sqrt(grad_x_gt[:-1, :, :] ** 2 + grad_y_gt[:, :-1, :] ** 2).mean())

    # 位移幅度统计
    pred_mag = np.sqrt(pred[:, :, 0] ** 2 + pred[:, :, 1] ** 2)
    gt_mag = np.sqrt(target[:, :, 0] ** 2 + target[:, :, 1] ** 2)
    if valid.sum() > 0:
        metrics['mean_magnitude_pred'] = float(pred_mag[valid].mean())
        metrics['mean_magnitude_gt'] = float(gt_mag[valid].mean())

    return metrics


# ===================== 评估主流程 =====================
@torch.no_grad()
def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"引导采样: {'开启' if not args.no_guidance else '关闭'}")

    # 加载模型
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
    print(f"模型: {ckpt_path.name} (epoch {ckpt.get('epoch', '?')})")

    diffusion = DiffusionSchedule().to(device)

    # 数据
    pairs = collect_pairs()
    train_pairs, val_pairs, test_pairs = split_data(pairs, seed=SEED)
    if args.split == 'val':
        eval_pairs = val_pairs
    elif args.split == 'train':
        eval_pairs = train_pairs
    else:
        eval_pairs = test_pairs
    print(f"评估集: {args.split} ({len(eval_pairs)} 个样本)\n")

    # 逐样本评估
    all_metrics = []
    total_time = 0.0
    results_dir = RESULTS_DIR / f"eval_{args.split}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for i, info in enumerate(eval_pairs):
        vertices, faces = load_model(info['shape_path'])
        if vertices is None or len(faces) == 0:
            continue
        vertices[:, 2] -= vertices[:, 2].min()
        enc = compute_layered_depth(vertices, faces, IMG_SIZE, K_LAYERS)

        data = np.load(info['map_path'])
        gt_disp = data['output_grid'] - data['input_grid']
        valid_mask = data['valid_mask'].astype(np.float32)
        gt_disp[~data['valid_mask']] = 0.0  # 无效光线位移置0, 避免 scipy resize 时 NaN 扩散污染有效像素
        disp_max = np.abs(gt_disp[data['valid_mask']]).max() if data['valid_mask'].sum() > 0 else 1.0
        disp_max = max(float(disp_max), 1e-6)
        gt_norm = gt_disp / disp_max

        mask_r = _resize_2d(valid_mask, IMG_SIZE)
        gt_r = _resize_2d(gt_norm, IMG_SIZE)
        from scipy.ndimage import uniform_filter
        conf_r = np.clip(uniform_filter(mask_r, size=3) * 1.5, 0, 1).astype(np.float32)

        cond_np = np.concatenate([enc, mask_r[:, :, None], conf_r[:, :, None]], axis=-1)
        cond = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        known = torch.from_numpy(gt_r).permute(2, 0, 1).unsqueeze(0).float().to(device)
        known_m = torch.from_numpy(mask_r).unsqueeze(0).unsqueeze(0).float().to(device)

        # 推理计时
        t0 = time.time()
        if args.no_guidance:
            pred = diffusion.sample(model, (1, CHANNELS_OUT, IMG_SIZE, IMG_SIZE), cond, device)
        else:
            pred = diffusion.sample_guided(model, (1, CHANNELS_OUT, IMG_SIZE, IMG_SIZE),
                                           cond, known, known_m, GUIDANCE_STRENGTH, device)
        infer_time = time.time() - t0
        total_time += infer_time

        pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)

        # 计算指标
        metrics = compute_metrics(pred_np, gt_r, mask_r)
        metrics['name'] = info['name']
        metrics['category'] = info['category']
        metrics['infer_time'] = infer_time
        metrics['valid_ratio'] = float(mask_r.mean())
        all_metrics.append(metrics)

        # 保存预测
        np.savez_compressed(
            results_dir / f"{info['name']}_eval.npz",
            predicted=pred_np, gt=gt_r, mask=mask_r,
            predicted_physical=pred_np * disp_max,
        )
        print(f"  [{i+1}/{len(eval_pairs)}] {info['name']:12s} | "
              f"RMSE={metrics.get('rmse_valid', 0):.4f} | "
              f"SSIM={metrics.get('ssim_valid', 0):.4f} | "
              f"{infer_time:.1f}s")

    # 汇总统计
    if not all_metrics:
        print("无有效评估样本")
        return

    summary = {
        'split': args.split,
        'n_samples': len(all_metrics),
        'guidance': not args.no_guidance,
        'avg_infer_time': total_time / len(all_metrics),
        'total_time': total_time,
        'aggregate': {},
        'per_sample': all_metrics,
    }

    # 聚合指标
    for key in ['mse_valid', 'mae_valid', 'rmse_valid', 'ssim_valid',
                'max_error_valid', 'magnitude_error_valid',
                'mse_missing', 'smoothness_pred', 'smoothness_gt',
                'mean_magnitude_pred', 'mean_magnitude_gt', 'valid_ratio']:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            summary['aggregate'][key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }

    # 打印报告
    print(f"\n{'='*60}")
    print(f"  评估报告 ({args.split} 集, {len(all_metrics)} 样本)")
    print(f"{'='*60}")
    agg = summary['aggregate']
    if 'rmse_valid' in agg:
        print(f"  有效区域 RMSE:  {agg['rmse_valid']['mean']:.5f} ± {agg['rmse_valid']['std']:.5f}")
    if 'mae_valid' in agg:
        print(f"  有效区域 MAE:   {agg['mae_valid']['mean']:.5f} ± {agg['mae_valid']['std']:.5f}")
    if 'ssim_valid' in agg:
        print(f"  有效区域 SSIM:  {agg['ssim_valid']['mean']:.4f} ± {agg['ssim_valid']['std']:.4f}")
    if 'mse_missing' in agg:
        print(f"  缺失区域 MSE:   {agg['mse_missing']['mean']:.5f} (补全参考)")
    if 'smoothness_pred' in agg:
        print(f"  预测平滑度:     {agg['smoothness_pred']['mean']:.5f}")
        print(f"  GT 平滑度:      {agg['smoothness_gt']['mean']:.5f}")
    print(f"  平均推理时间:   {summary['avg_infer_time']:.2f}s / 样本")
    print(f"  平均有效率:     {agg.get('valid_ratio', {}).get('mean', 0)*100:.1f}%")
    print(f"{'='*60}")

    # 保存报告
    report_path = results_dir / "eval_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {report_path}")
    print(f"预测: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="棱镜扩散模型 - 测试评估")
    parser.add_argument("--split", choices=['test', 'val', 'train'], default='test',
                        help="评估哪个数据集划分")
    parser.add_argument("--no_guidance", action="store_true",
                        help="关闭引导采样 (纯生成)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_evaluation(args)


if __name__ == "__main__":
    main()
