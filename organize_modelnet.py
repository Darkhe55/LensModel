"""
整理 ModelNet40 解压出的原始类别文件夹
======================================
将直接解压到 shapes/ 下的原始英文类别文件夹 (airplane, bathtub, ... xbox,
其 .off 文件嵌套在 train/test 子目录) 转换整理为项目统一的编号中文 PLY 文件夹
(17_多面体_飞机, 18_多面体_浴缸, ... 56_多面体_游戏机)，与现有分类风格一致，
并自动纳入 shape2map 的处理范围。

功能:
  - 递归读取原始文件夹下所有 .off (含 train/test 子目录)
  - OFF → PLY 转换, 复用 download_models.py 的解析器
  - 每类补齐到目标数量 (默认 ~8000/40 ≈ 200), 已存在的 PLY 自动跳过
  - 可选删除整理完成的原始文件夹 (--remove_raw)

运行 (Comfyui 环境):
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe organize_modelnet.py
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe organize_modelnet.py --target 8000
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe organize_modelnet.py --remove_raw   # 转换后删除原始文件夹
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_models import parse_off, write_ply, MODELNET_CATEGORIES, BASE_DIR


def organize(target=8000, remove_raw=False):
    """将原始 ModelNet 类别文件夹整理为编号 PLY 文件夹"""
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    per_cat = max(1, target // len(MODELNET_CATEGORIES))
    print(f"整理 ModelNet 原始文件夹 → 编号 PLY 文件夹")
    print(f"  目标总数 ~{target}, 每类 ~{per_cat} 个, 共 {len(MODELNET_CATEGORIES)} 类")
    print(f"{'='*55}")

    total_new, total_have, raw_found = 0, 0, 0
    for i, (eng, chn) in enumerate(MODELNET_CATEGORIES):
        src = BASE_DIR / eng                 # shapes/airplane (原始文件夹)
        dst = BASE_DIR / f"{17 + i}_多面体_{chn}"  # shapes/17_多面体_飞机 (目标)

        if not src.exists():
            continue  # 该类别原始文件夹不存在, 跳过
        raw_found += 1
        dst.mkdir(parents=True, exist_ok=True)

        # 递归收集所有 .off (含 train/test 子目录)
        off_files = sorted(src.rglob("*.off"))
        # 已有 PLY (跳过, 但计入配额)
        existing = {p.stem for p in dst.glob("*.ply")}
        n_have = len(existing)
        need = max(0, per_cat - n_have)

        new = 0
        for off in off_files:
            if new >= need:
                break
            if off.stem in existing:
                continue
            try:
                text = off.read_text(encoding='utf-8', errors='ignore')
                verts, faces = parse_off(text)
                if len(verts) < 3 or len(faces) < 1:
                    continue
                write_ply(dst / f"{off.stem}.ply", verts, faces)
                new += 1
            except Exception as e:
                print(f"    [FAIL] {off.name}: {e}")

        total_new += new
        total_have += n_have
        print(f"  {dst.name}: 新增 {new}, 已有 {n_have} (合计 {n_have + new})")

    print(f"{'='*55}")
    print(f"  整理完成: 新增 {total_new} 个 PLY, 原有 {total_have} 个, 涉及 {raw_found} 个原始文件夹")

    # 可选: 删除已整理的原始文件夹
    if remove_raw and raw_found > 0:
        import shutil
        print(f"\n  清理原始文件夹 ...")
        for eng, chn in MODELNET_CATEGORIES:
            src = BASE_DIR / eng
            if src.exists():
                shutil.rmtree(src)
                print(f"    已删除: {eng}/")
        print(f"  原始文件夹清理完成")
    elif raw_found > 0:
        print(f"\n  提示: 原始文件夹仍保留, 加 --remove_raw 可在转换后删除")

    # 统计 shapes 下编号文件夹总数
    grand = 0
    for sub in sorted(BASE_DIR.iterdir()):
        if sub.is_dir() and not sub.name.startswith('_') and sub.name[0:2].isdigit():
            grand += len([f for f in sub.iterdir() if f.suffix.lower() in ('.ply', '.obj')])
    print(f"\n  shapes 编号文件夹模型总数: {grand}")


def main():
    parser = argparse.ArgumentParser(description="整理 ModelNet 原始文件夹为编号 PLY 文件夹")
    parser.add_argument("--target", type=int, default=8000, help="目标总数 (默认 8000)")
    parser.add_argument("--remove_raw", action="store_true", help="转换后删除原始英文文件夹")
    args = parser.parse_args()
    organize(target=args.target, remove_raw=args.remove_raw)


if __name__ == "__main__":
    main()

