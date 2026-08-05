"""
棱镜模型 - 可视化工具
======================
将位移场、编码、评估结果渲染为图像。

功能:
  1. 位移场可视化 (箭头图 / 颜色编码)
  2. 多层深度编码可视化
  3. GT vs 预测对比图
  4. 误差热力图
  5. 批量生成评估图板

运行:
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe visualize_diffusion.py results/eval_test/正四面体_eval.npz
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe visualize.py --batch results/eval_test/
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe visualize.py --encoding shapes/01_正多面体/正四面体.ply
"""
import os, sys, argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 配置中文字体 (Windows 微软雅黑/黑体), 避免中文显示为方框或缺字形警告
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def plot_displacement_field(disp, mask=None, title="位移场", ax=None, quiver_skip=4):
    """绘制位移场 (箭头 + 颜色编码幅度)"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    H, W = disp.shape[:2]
    mag = np.sqrt(disp[:, :, 0] ** 2 + disp[:, :, 1] ** 2)

    # 背景: 幅度颜色
    im = ax.imshow(mag, cmap='hot', interpolation='bilinear', alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, label='位移幅度')

    # 箭头
    y, x = np.mgrid[0:H:quiver_skip, 0:W:quiver_skip]
    u = disp[::quiver_skip, ::quiver_skip, 0]
    v = disp[::quiver_skip, ::quiver_skip, 1]
    ax.quiver(x, y, u, -v, color='cyan', alpha=0.6, scale=15, width=0.003)

    if mask is not None:
        ax.contour(mask, levels=[0.5], colors='lime', linewidths=0.8, alpha=0.7)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax


def plot_encoding(encoding, title="多层深度编码"):
    """可视化多层深度编码的各通道"""
    n_ch = encoding.shape[2]
    cols = min(4, n_ch)
    rows = (n_ch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    ch_names = [f'深度层 {i}' for i in range(K_LAYERS)] + ['厚度', '占据掩码', 'valid', '置信度']
    for i in range(n_ch):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        im = ax.imshow(encoding[:, :, i], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(ch_names[i] if i < len(ch_names) else f'ch{i}', fontsize=8)
        ax.axis('off')
    # 隐藏多余子图
    for i in range(n_ch, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_comparison(gt, pred, mask=None, name=""):
    """GT vs 预测对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # GT
    plot_displacement_field(gt, mask, "GT 位移场", axes[0, 0])
    # 预测
    plot_displacement_field(pred, mask, "预测位移场", axes[0, 1])
    # 误差幅度
    err = np.sqrt((pred - gt)[:, :, 0] ** 2 + (pred - gt)[:, :, 1] ** 2)
    im = axes[0, 2].imshow(err, cmap='Reds', interpolation='bilinear')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    axes[0, 2].set_title('误差幅度', fontsize=10)

    # GT dx
    axes[1, 0].imshow(gt[:, :, 0], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('GT dx', fontsize=10)
    # Pred dx
    axes[1, 1].imshow(pred[:, :, 0], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('预测 dx', fontsize=10)
    # dx 误差
    axes[1, 2].imshow(np.abs(pred[:, :, 0] - gt[:, :, 0]), cmap='Oranges')
    axes[1, 2].set_title('|dx 误差|', fontsize=10)

    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle(f'光路映射对比: {name}', fontsize=13)
    plt.tight_layout()
    return fig


def plot_error_histogram(gt, pred, mask, name=""):
    """误差分布直方图"""
    valid = mask > 0.5
    if valid.sum() == 0:
        return None
    err_x = (pred[:, :, 0] - gt[:, :, 0])[valid]
    err_y = (pred[:, :, 1] - gt[:, :, 1])[valid]
    err_mag = np.sqrt(err_x ** 2 + err_y ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(err_x, bins=50, color='steelblue', alpha=0.7, edgecolor='none')
    axes[0].set_title('dx 误差分布')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=0.8)
    axes[1].hist(err_y, bins=50, color='coral', alpha=0.7, edgecolor='none')
    axes[1].set_title('dy 误差分布')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=0.8)
    axes[2].hist(err_mag, bins=50, color='mediumpurple', alpha=0.7, edgecolor='none')
    axes[2].set_title('误差幅度分布')
    axes[2].axvline(err_mag.mean(), color='red', linestyle='--', linewidth=0.8,
                    label=f'均值={err_mag.mean():.4f}')
    axes[2].legend()
    fig.suptitle(f'误差统计: {name}', fontsize=11)
    plt.tight_layout()
    return fig


# ===================== 批量处理 =====================
def visualize_single(npz_path, output_dir=None):
    """可视化单个评估结果"""
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem.replace('_eval', '').replace('_prediction', '')

    pred = data.get('predicted', data.get('predicted_normalized'))
    gt = data.get('gt', data.get('gt_displacement'))
    mask = data.get('mask', data.get('valid_mask'))

    if pred is None:
        print(f"  跳过 {npz_path.name}: 无预测数据")
        return

    # 对比图
    if gt is not None:
        fig = plot_comparison(gt, pred, mask, stem)
        fig.savefig(output_dir / f"{stem}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig = plot_error_histogram(gt, pred, mask, stem)
        if fig:
            fig.savefig(output_dir / f"{stem}_error_hist.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    # 单独位移场
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_displacement_field(pred, mask, f"预测: {stem}", ax)
    fig.savefig(output_dir / f"{stem}_pred_field.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [OK] {stem}")


def visualize_encoding(shape_path, output_dir=None):
    """可视化形状的多层深度编码"""
    from train_diffusion import compute_layered_depth, load_model
    shape_path = Path(shape_path)
    vertices, faces = load_model(shape_path)
    if vertices is None:
        print(f"错误: 无法加载 {shape_path}")
        return
    vertices[:, 2] -= vertices[:, 2].min()
    enc = compute_layered_depth(vertices, faces, IMG_SIZE, K_LAYERS)

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_encoding(enc, f"多层深度编码: {shape_path.stem}")
    fig.savefig(output_dir / f"{shape_path.stem}_encoding.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {shape_path.stem} 编码可视化")


def visualize_batch(results_dir, output_dir=None):
    """批量可视化评估结果"""
    results_dir = Path(results_dir)
    npz_files = sorted(results_dir.glob("*_eval.npz")) + sorted(results_dir.glob("*_prediction.npz"))
    if not npz_files:
        print(f"无 NPZ 文件: {results_dir}")
        return
    print(f"找到 {len(npz_files)} 个结果文件")
    for f in npz_files:
        visualize_single(f, output_dir)


def main():
    parser = argparse.ArgumentParser(description="棱镜光路映射 - 可视化")
    parser.add_argument("input", nargs="?", type=str, default=None,
                        help="NPZ 文件路径或结果目录")
    parser.add_argument("--batch", type=str, default=None, help="批量可视化目录")
    parser.add_argument("--encoding", type=str, default=None, help="可视化形状编码")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出图片目录")
    args = parser.parse_args()

    if args.encoding:
        visualize_encoding(args.encoding, args.output)
    elif args.batch:
        visualize_batch(args.batch, args.output)
    elif args.input:
        p = Path(args.input)
        if p.is_dir():
            visualize_batch(p, args.output)
        else:
            visualize_single(p, args.output)
    else:
        # 默认: 可视化 results/eval_test/
        default_dir = RESULTS_DIR / "eval_test"
        if default_dir.exists():
            visualize_batch(default_dir, args.output)
        else:
            print("请指定输入文件或目录。用法: python visualize.py <path>")


if __name__ == "__main__":
    main()
