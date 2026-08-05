"""
从开源模型库下载示例模型到 shapes 文件夹并分类
=================================================
来源:
  1. FSU PLY Database (GNU LGPL): https://people.sc.fsu.edu/~jburkardt/data/ply/
  2. GitHub common-3d-test-models: https://github.com/alecjacobson/common-3d-test-models
  3. Princeton ModelNet40 (CAD 多面体, OFF 格式): 40 类共 12311 个模型
     https://modelnet.cs.princeton.edu/ModelNet40.zip

功能:
  - 精选模型下载 (FSU/GitHub, 已有则跳过)
  - ModelNet40 批量下载 (约 8000 个多面体, OFF→PLY 转换, 已有则跳过)
  - 按现有风格分类: 17_多面体_飞机, 18_多面体_浴缸, ...

运行 (Comfyui 环境):
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe download_models.py
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe download_models.py --target 8000
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe download_models.py --no_modelnet      # 仅精选
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe download_models.py --no_curated      # 仅ModelNet
  C:\\Users\\MerchRev\\.conda\\envs\\Comfyui\\python.exe download_models.py --keep_zip        # 保留zip
"""
import os, sys, time, zipfile, argparse, urllib.request, ssl
from pathlib import Path

BASE_DIR = Path(__file__).parent / "shapes"
CACHE_DIR = BASE_DIR / "_cache"

# 忽略 SSL 验证（某些学术站点证书不全）
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

FSU_BASE = "https://people.sc.fsu.edu/~jburkardt/data/ply/"
GH_BASE = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"
MODELNET40_URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"

# ============ 精选下载列表 (分类 -> [(文件名, 源URL, 保存名)]) ============
MODELS = {
    "11_几何体_开源": [
        ("cube.ply",           FSU_BASE + "cube.ply",           "立方体.ply"),
        ("dodecahedron.ply",   FSU_BASE + "dodecahedron.ply",   "正十二面体.ply"),
        ("icosahedron.ply",    FSU_BASE + "icosahedron.ply",    "正二十面体.ply"),
        ("octahedron.ply",     FSU_BASE + "octahedron.ply",     "正八面体.ply"),
        ("pyramid.ply",        FSU_BASE + "pyramid.ply",        "金字塔.ply"),
        ("sphere.ply",         FSU_BASE + "sphere.ply",         "球体.ply"),
        ("tetrahedron.ply",    FSU_BASE + "tetrahedron.ply",    "正四面体.ply"),
        ("helix.ply",          FSU_BASE + "helix.ply",          "螺旋体.ply"),
    ],
    "12_动物_开源": [
        ("ant.ply",            FSU_BASE + "ant.ply",            "蚂蚁.ply"),
        ("cow.ply",            FSU_BASE + "cow.ply",            "奶牛.ply"),
        ("dolphins.ply",       FSU_BASE + "dolphins.ply",       "海豚.ply"),
        ("egret.ply",          FSU_BASE + "egret.ply",          "白鹭.ply"),
        ("shark.ply",          FSU_BASE + "shark.ply",          "鲨鱼.ply"),
        ("big_spider.ply",     FSU_BASE + "big_spider.ply",     "蜘蛛.ply"),
        ("hammerhead.ply",     FSU_BASE + "hammerhead.ply",     "锤头鲨.ply"),
        ("hind.ply",           FSU_BASE + "hind.ply",           "雌鹿.ply"),
        ("armadillo.obj",      GH_BASE + "armadillo.obj",       "犰狳.obj"),
        ("horse.obj",          GH_BASE + "horse.obj",           "马.obj"),
        ("bunny.obj",          GH_BASE + "bunny.obj",           "斯坦福兔子.obj"),
        ("spot.obj",           GH_BASE + "spot.obj",            "斑点狗.obj"),
        ("alligator.obj",      GH_BASE + "alligator.obj",       "鳄鱼.obj"),
    ],
    "13_交通工具_开源": [
        ("airplane.ply",       FSU_BASE + "airplane.ply",       "飞机.ply"),
        ("f16.ply",            FSU_BASE + "f16.ply",            "F16战斗机.ply"),
        ("chopper.ply",        FSU_BASE + "chopper.ply",        "直升机.ply"),
        ("big_porsche.ply",    FSU_BASE + "big_porsche.ply",    "保时捷.ply"),
        ("big_dodge.ply",      FSU_BASE + "big_dodge.ply",      "道奇跑车.ply"),
        ("pickup_big.ply",     FSU_BASE + "pickup_big.ply",     "皮卡车.ply"),
        ("galleon.ply",        FSU_BASE + "galleon.ply",        "帆船.ply"),
        ("saratoga.ply",       FSU_BASE + "saratoga.ply",       "萨拉托加号.ply"),
    ],
    "14_人物雕塑_开源": [
        ("beethoven.ply",      FSU_BASE + "beethoven.ply",      "贝多芬像.ply"),
        ("head1.ply",          FSU_BASE + "head1.ply",          "人头扫描1.ply"),
        ("head2.ply",          FSU_BASE + "head2.ply",          "人头扫描2.ply"),
        ("skull.ply",          FSU_BASE + "skull.ply",          "骷髅头.ply"),
        ("footbones.ply",      FSU_BASE + "footbones.ply",      "足骨.ply"),
        ("happy_buddha.obj",   GH_BASE + "happy.obj",           "弥勒佛.obj"),
        ("ogre.obj",           GH_BASE + "ogre.obj",            "食人魔.obj"),
        ("suzanne.obj",        GH_BASE + "suzanne.obj",         "Suzanne猴头.obj"),
    ],
    "15_日常物品_开源": [
        ("mug.ply",            FSU_BASE + "mug.ply",            "马克杯.ply"),
        ("teapot.ply",         FSU_BASE + "teapot.ply",         "茶壶.ply"),
        ("ketchup.ply",        FSU_BASE + "ketchup.ply",        "番茄酱瓶.ply"),
        ("trashcan.ply",       FSU_BASE + "trashcan.ply",       "垃圾桶.ply"),
        ("urn2.ply",           FSU_BASE + "urn2.ply",           "花瓶.ply"),
        ("walkman.ply",        FSU_BASE + "walkman.ply",        "随身听.ply"),
        ("tennis_shoe.ply",    FSU_BASE + "tennis_shoe.ply",    "运动鞋.ply"),
        ("sandal.ply",         FSU_BASE + "sandal.ply",         "凉鞋.ply"),
        ("steeringweel.ply",   FSU_BASE + "steeringweel.ply",   "方向盘.ply"),
        ("street_lamp.ply",    FSU_BASE + "street_lamp.ply",    "路灯.ply"),
        ("kerolamp.ply",       FSU_BASE + "kerolamp.ply",       "煤油灯.ply"),
        ("apple.ply",          FSU_BASE + "apple.ply",          "苹果.ply"),
        ("scissors.ply",       FSU_BASE + "scissors.ply",       "剪刀.ply"),
        ("stratocaster.ply",   FSU_BASE + "stratocaster.ply",   "电吉他.ply"),
    ],
    "16_机械零件_开源": [
        ("part.ply",           FSU_BASE + "part.ply",           "机械零件.ply"),
        ("pump.ply",           FSU_BASE + "pump.ply",           "水泵.ply"),
        ("turbine.ply",        FSU_BASE + "turbine.ply",        "涡轮.ply"),
        ("balance.ply",        FSU_BASE + "balance.ply",        "天平.ply"),
        ("weathervane.ply",    FSU_BASE + "weathervane.ply",    "风向标.ply"),
        ("canstick.ply",       FSU_BASE + "canstick.ply",       "罐装棒.ply"),
        ("dart.ply",           FSU_BASE + "dart.ply",           "飞镖.ply"),
        ("tommygun.ply",       FSU_BASE + "tommygun.ply",       "汤姆逊冲锋枪.ply"),
        ("fracttree.ply",      FSU_BASE + "fracttree.ply",      "分形树.ply"),
    ],
}

# ============ ModelNet40 类别 (英文 -> 中文), 文件夹编号从 17 开始 ============
MODELNET_CATEGORIES = [
    ("airplane", "飞机"), ("bathtub", "浴缸"), ("bed", "床"), ("bench", "长凳"),
    ("bookshelf", "书架"), ("bottle", "瓶子"), ("bowl", "碗"), ("car", "汽车"),
    ("chair", "椅子"), ("cone", "圆锥"), ("cup", "杯子"), ("curtain", "窗帘"),
    ("desk", "书桌"), ("door", "门"), ("dresser", "梳妆台"), ("flower_pot", "花盆"),
    ("glass_box", "玻璃盒"), ("guitar", "吉他"), ("keyboard", "键盘"), ("lamp", "台灯"),
    ("laptop", "笔记本电脑"), ("mantel", "壁炉架"), ("monitor", "显示器"), ("night_stand", "床头柜"),
    ("person", "人"), ("piano", "钢琴"), ("plant", "植物"), ("radio", "收音机"),
    ("range_hood", "油烟机"), ("sink", "水槽"), ("sofa", "沙发"), ("stairs", "楼梯"),
    ("stool", "凳子"), ("table", "桌子"), ("tent", "帐篷"), ("toilet", "马桶"),
    ("tv_stand", "电视柜"), ("vase", "花瓶"), ("wardrobe", "衣柜"), ("xbox", "游戏机"),
]
_CAT_SET = {eng for eng, _ in MODELNET_CATEGORIES}


# ===================== 下载工具 =====================
def _github_to_jsdelivr(url):
    """将 raw.githubusercontent.com 链接转为 jsdelivr CDN 镜像 (国内加速)"""
    prefix = "https://raw.githubusercontent.com/"
    if url.startswith(prefix):
        parts = url[len(prefix):].split('/', 3)  # user/repo/branch/path
        if len(parts) == 4:
            user, repo, branch, path = parts
            return f"https://cdn.jsdelivr.net/gh/{user}/{repo}@{branch}/{path}"
    return None


def _candidate_urls(url):
    """返回候选 URL 列表: GitHub 文件优先用 jsdelivr 镜像, 原始链接兑底"""
    mirror = _github_to_jsdelivr(url)
    if mirror:
        return [mirror, url]
    return [url]


def download_file(url, dest, retries=3):
    """下载单个小文件，带重试 + GitHub 镜像自动回退"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_urls(url)
    last_err = None
    for attempt in range(retries):
        for cand in candidates:
            try:
                req = urllib.request.Request(cand, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                    data = resp.read()
                if len(data) < 100:
                    raise ValueError(f"文件过小({len(data)}B)，可能无效")
                with open(dest, 'wb') as f:
                    f.write(data)
                return len(data)
            except Exception as e:
                last_err = e
                continue  # 尝试下一个候选 URL
        time.sleep(1)
    print(f"    [FAIL] {url} -> {last_err}")
    return -1


def download_large(url, dest, retries=100, chunk=1024 * 256, idle_timeout=60):
    """下载大文件, 支持 HTTP Range 断点续传 + 大量重试。
    网络中断后从已下载位置继续, 而非从头重来; 直到下载完整或达到最大尝试次数。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=idle_timeout, context=ssl_ctx) as resp:
                code = resp.getcode()
                # 判断服务器是否接受续传
                if existing > 0 and code == 206:
                    mode = 'ab'      # 续传: 追加
                    downloaded = existing
                else:
                    mode = 'wb'      # 服务器不支持 Range 或全新下载: 覆盖
                    downloaded = 0

                # 计算总大小
                crange = resp.headers.get('Content-Range')
                if crange and '/' in crange:
                    total = int(crange.split('/')[-1])
                else:
                    total = downloaded + int(resp.headers.get('Content-Length', 0))

                with open(dest, mode) as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        if total:
                            print(f"\r  下载中: {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB "
                                  f"({downloaded/total*100:.1f}%) [第{attempt}次连接]", end="", flush=True)
                print()
                # 下载完成判断
                if total and downloaded >= total:
                    print(f"  下载完成: {downloaded/1024/1024:.1f} MB")
                    return True
                print(f"  连接中断于 {downloaded/1024/1024:.1f} MB, 准备续传...")
        except Exception as e:
            cur = dest.stat().st_size if dest.exists() else 0
            print(f"\n  [第{attempt}次连接] {type(e).__name__}: {e} (已缓存 {cur/1024/1024:.1f} MB)")
        time.sleep(2)

    # 达到最大尝试次数, 用 zip 完整性兑底判断
    ok = dest.exists() and zipfile.is_zipfile(dest)
    if not ok:
        print(f"  [ERROR] 达到最大重试次数仍未下载完整, 已保留部分文件, 重新运行可继续续传")
    return ok


# ===================== OFF → PLY 转换 =====================
def parse_off(text):
    """解析 OFF 格式文本, 返回 (vertices, faces) 并三角化面片"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith('#')]
    if not lines:
        return [], []
    idx = 0
    header = lines[0].upper()
    if header.startswith('OFF'):
        rest = lines[0][3:].strip()
        if rest:  # 计数与 OFF 同行
            counts = rest.split()
            n_v, n_f = int(counts[0]), int(counts[1])
            idx = 1
        else:
            counts = lines[1].split()
            n_v, n_f = int(counts[0]), int(counts[1])
            idx = 2
    else:
        counts = lines[0].split()
        n_v, n_f = int(counts[0]), int(counts[1])
        idx = 1

    vertices = []
    for i in range(n_v):
        parts = lines[idx + i].split()
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    idx += n_v

    faces = []
    for i in range(n_f):
        parts = lines[idx + i].split()
        k = int(parts[0])
        fidx = [int(parts[j + 1]) for j in range(k)]
        for j in range(1, k - 1):  # 扇形三角化
            faces.append([fidx[0], fidx[j], fidx[j + 1]])
    return vertices, faces


def write_ply(filepath, vertices, faces):
    """写 ASCII PLY (与项目加载器兼容: x y z + 面片列表)"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# ===================== 精选模型下载 =====================
def download_curated():
    total_ok, total_fail, total_bytes = 0, 0, 0
    for folder, items in MODELS.items():
        print(f"\n{'='*50}\n  {folder} ({len(items)} 个模型)\n{'='*50}")
        for src_name, url, save_name in items:
            dest = BASE_DIR / folder / save_name
            if dest.exists():
                print(f"  [SKIP] {save_name} (已存在)")
                total_ok += 1
                continue
            size = download_file(url, dest)
            if size > 0:
                print(f"  [OK] {save_name}  ({size/1024:.1f} KB)")
                total_ok += 1
                total_bytes += size
            else:
                total_fail += 1
            time.sleep(0.3)
    print(f"\n  精选下载: 成功 {total_ok}, 失败 {total_fail}, {total_bytes/1024/1024:.2f} MB")
    return total_ok, total_fail


# ===================== ModelNet40 批量下载 =====================
def download_modelnet40(target=8000, keep_zip=False):
    """下载 ModelNet40, 按类别转换为 PLY, 总数约 target 个"""
    zip_path = CACHE_DIR / "ModelNet40.zip"

    # 1) 下载 zip (支持断点续传; 已完整则跳过)
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        print(f"\n  [SKIP] ModelNet40.zip 已完整 ({zip_path.stat().st_size/1024/1024:.1f} MB)")
    else:
        if zip_path.exists():
            print(f"\n  检测到未完成下载 ({zip_path.stat().st_size/1024/1024:.1f} MB), 断点续传中...")
        else:
            print(f"\n  下载 ModelNet40.zip (~1.9 GB), 支持断点续传 ...")
        if not download_large(MODELNET40_URL, zip_path):
            print("  [ERROR] ModelNet40.zip 下载失败 (已保留部分文件, 重新运行可续传)")
            return 0, 0
    if not zipfile.is_zipfile(zip_path):
        print("  [ERROR] zip 文件损坏，请删除后重试:", zip_path)
        return 0, 0

    # 2) 按类别选择性读取 + 转换
    per_cat = max(1, target // len(MODELNET_CATEGORIES))
    print(f"\n  目标总数 ~{target}, 每类 ~{per_cat} 个, 共 {len(MODELNET_CATEGORIES)} 类")
    print(f"{'='*50}")

    total_ok, total_skip = 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for i, (eng, chn) in enumerate(MODELNET_CATEGORIES):
            folder = BASE_DIR / f"{17 + i}_多面体_{chn}"
            folder.mkdir(parents=True, exist_ok=True)

            # 该类别的所有 .off 文件
            cat_files = sorted(n for n in names
                               if n.lower().endswith('.off') and eng in n.replace('\\', '/').split('/'))
            count, converted, skipped = 0, 0, 0
            for n in cat_files:
                if count >= per_cat:
                    break
                stem = Path(n).stem
                ply_path = folder / f"{stem}.ply"
                if ply_path.exists():  # 已有则跳过, 但计入配额
                    count += 1
                    skipped += 1
                    continue
                try:
                    off_text = zf.read(n).decode('utf-8', errors='ignore')
                    verts, faces = parse_off(off_text)
                    if len(verts) < 3 or len(faces) < 1:
                        continue
                    write_ply(ply_path, verts, faces)
                    count += 1
                    converted += 1
                except Exception as e:
                    print(f"    [FAIL] {n}: {e}")
            total_ok += converted
            total_skip += skipped
            print(f"  {folder.name}: 新增 {converted}, 已有 {skipped} (共 {count})")

    # 3) 清理 zip
    if not keep_zip:
        try:
            zip_path.unlink()
            print(f"\n  已清理 zip: {zip_path.name} (用 --keep_zip 可保留)")
        except Exception:
            pass

    print(f"\n  ModelNet40: 新增 {total_ok}, 跳过已有 {total_skip}")
    return total_ok, total_skip


# ===================== 统计 =====================
def print_summary():
    print(f"\n{'='*50}\n  shapes 文件夹统计\n{'='*50}")
    grand = 0
    for sub in sorted(BASE_DIR.iterdir()):
        if sub.is_dir() and not sub.name.startswith('_'):
            n = len([f for f in sub.iterdir() if f.is_file()])
            grand += n
            print(f"  {sub.name}: {n}")
    print(f"{'='*50}\n  总计: {grand} 个模型\n{'='*50}")


def main():
    # Windows 控制台中文输出
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="开源多面体模型批量下载")
    parser.add_argument("--target", type=int, default=8000, help="ModelNet40 目标总数 (默认 8000)")
    parser.add_argument("--no_curated", action="store_true", help="跳过精选模型下载")
    parser.add_argument("--no_modelnet", action="store_true", help="跳过 ModelNet40 批量下载")
    parser.add_argument("--keep_zip", action="store_true", help="转换后保留 ModelNet40.zip")
    args = parser.parse_args()

    print(f"开源模型下载器")
    print(f"  保存路径: {BASE_DIR}")
    print(f"  ModelNet 目标: {args.target}")

    if not args.no_curated:
        download_curated()
    if not args.no_modelnet:
        download_modelnet40(target=args.target, keep_zip=args.keep_zip)

    print_summary()


if __name__ == "__main__":
    main()
"""
从开源模型库下载示例模型到 shapes 文件夹并分类
来源:
  1. FSU PLY Database (GNU LGPL): https://people.sc.fsu.edu/~jburkardt/data/ply/
  2. GitHub common-3d-test-models: https://github.com/alecjacobson/common-3d-test-models
运行: C:\\ProgramData\\miniconda3\\envs\\Math\\python.exe download_models.py
"""
import os, sys, time, urllib.request, ssl
from pathlib import Path

BASE_DIR = Path(__file__).parent / "shapes"

# 忽略 SSL 验证（某些学术站点证书不全）
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

FSU_BASE = "https://people.sc.fsu.edu/~jburkardt/data/ply/"
GH_BASE = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"

# ============ 下载列表 (分类 -> [(文件名, 源URL, 保存名)]) ============
MODELS = {
    "11_几何体_开源": [
        ("cube.ply",           FSU_BASE + "cube.ply",           "立方体.ply"),
        ("dodecahedron.ply",   FSU_BASE + "dodecahedron.ply",   "正十二面体.ply"),
        ("icosahedron.ply",    FSU_BASE + "icosahedron.ply",    "正二十面体.ply"),
        ("octahedron.ply",     FSU_BASE + "octahedron.ply",     "正八面体.ply"),
        ("pyramid.ply",        FSU_BASE + "pyramid.ply",        "金字塔.ply"),
        ("sphere.ply",         FSU_BASE + "sphere.ply",         "球体.ply"),
        ("tetrahedron.ply",    FSU_BASE + "tetrahedron.ply",    "正四面体.ply"),
        ("helix.ply",          FSU_BASE + "helix.ply",          "螺旋体.ply"),
    ],
    "12_动物_开源": [
        ("ant.ply",            FSU_BASE + "ant.ply",            "蚂蚁.ply"),
        ("cow.ply",            FSU_BASE + "cow.ply",            "奶牛.ply"),
        ("dolphins.ply",       FSU_BASE + "dolphins.ply",       "海豚.ply"),
        ("egret.ply",          FSU_BASE + "egret.ply",          "白鹭.ply"),
        ("shark.ply",          FSU_BASE + "shark.ply",          "鲨鱼.ply"),
        ("big_spider.ply",     FSU_BASE + "big_spider.ply",     "蜘蛛.ply"),
        ("hammerhead.ply",     FSU_BASE + "hammerhead.ply",     "锤头鲨.ply"),
        ("hind.ply",           FSU_BASE + "hind.ply",           "雌鹿.ply"),
        # GitHub OBJ 模型
        ("armadillo.obj",      GH_BASE + "armadillo.obj",       "犰狳.obj"),
        ("horse.obj",          GH_BASE + "horse.obj",           "马.obj"),
        ("bunny.obj",          GH_BASE + "bunny.obj",           "斯坦福兔子.obj"),
        ("spot.obj",           GH_BASE + "spot.obj",            "斑点狗.obj"),
        ("alligator.obj",      GH_BASE + "alligator.obj",       "鳄鱼.obj"),
    ],
    "13_交通工具_开源": [
        ("airplane.ply",       FSU_BASE + "airplane.ply",       "飞机.ply"),
        ("f16.ply",            FSU_BASE + "f16.ply",            "F16战斗机.ply"),
        ("chopper.ply",        FSU_BASE + "chopper.ply",        "直升机.ply"),
        ("big_porsche.ply",    FSU_BASE + "big_porsche.ply",    "保时捷.ply"),
        ("big_dodge.ply",      FSU_BASE + "big_dodge.ply",      "道奇跑车.ply"),
        ("pickup_big.ply",     FSU_BASE + "pickup_big.ply",     "皮卡车.ply"),
        ("galleon.ply",        FSU_BASE + "galleon.ply",        "帆船.ply"),
        ("saratoga.ply",       FSU_BASE + "saratoga.ply",       "萨拉托加号.ply"),
    ],
    "14_人物雕塑_开源": [
        ("beethoven.ply",      FSU_BASE + "beethoven.ply",      "贝多芬像.ply"),
        ("head1.ply",          FSU_BASE + "head1.ply",          "人头扫描1.ply"),
        ("head2.ply",          FSU_BASE + "head2.ply",          "人头扫描2.ply"),
        ("skull.ply",          FSU_BASE + "skull.ply",          "骷髅头.ply"),
        ("footbones.ply",      FSU_BASE + "footbones.ply",      "足骨.ply"),
        # GitHub
        ("happy_buddha.obj",   GH_BASE + "happy.obj",           "弥勒佛.obj"),
        ("ogre.obj",           GH_BASE + "ogre.obj",            "食人魔.obj"),
        ("suzanne.obj",        GH_BASE + "suzanne.obj",         "Suzanne猴头.obj"),
    ],
    "15_日常物品_开源": [
        ("mug.ply",            FSU_BASE + "mug.ply",            "马克杯.ply"),
        ("teapot.ply",         FSU_BASE + "teapot.ply",         "茶壶.ply"),
        ("ketchup.ply",        FSU_BASE + "ketchup.ply",        "番茄酱瓶.ply"),
        ("trashcan.ply",       FSU_BASE + "trashcan.ply",       "垃圾桶.ply"),
        ("urn2.ply",           FSU_BASE + "urn2.ply",           "花瓶.ply"),
        ("walkman.ply",        FSU_BASE + "walkman.ply",        "随身听.ply"),
        ("tennis_shoe.ply",    FSU_BASE + "tennis_shoe.ply",    "运动鞋.ply"),
        ("sandal.ply",         FSU_BASE + "sandal.ply",         "凉鞋.ply"),
        ("steeringweel.ply",   FSU_BASE + "steeringweel.ply",   "方向盘.ply"),
        ("street_lamp.ply",    FSU_BASE + "street_lamp.ply",    "路灯.ply"),
        ("kerolamp.ply",       FSU_BASE + "kerolamp.ply",       "煤油灯.ply"),
        ("apple.ply",          FSU_BASE + "apple.ply",          "苹果.ply"),
        ("scissors.ply",       FSU_BASE + "scissors.ply",       "剪刀.ply"),
        ("stratocaster.ply",   FSU_BASE + "stratocaster.ply",   "电吉他.ply"),
    ],
    "16_机械零件_开源": [
        ("part.ply",           FSU_BASE + "part.ply",           "机械零件.ply"),
        ("pump.ply",           FSU_BASE + "pump.ply",           "水泵.ply"),
        ("turbine.ply",        FSU_BASE + "turbine.ply",        "涡轮.ply"),
        ("balance.ply",        FSU_BASE + "balance.ply",        "天平.ply"),
        ("weathervane.ply",    FSU_BASE + "weathervane.ply",    "风向标.ply"),
        ("canstick.ply",       FSU_BASE + "canstick.ply",       "罐装棒.ply"),
        ("dart.ply",           FSU_BASE + "dart.ply",           "飞镖.ply"),
        ("tommygun.ply",       FSU_BASE + "tommygun.ply",       "汤姆逊冲锋枪.ply"),
        ("fracttree.ply",      FSU_BASE + "fracttree.ply",      "分形树.ply"),
    ],
}


def download_file(url, dest, retries=3):
    """下载单个文件，带重试"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                data = resp.read()
            if len(data) < 100:
                raise ValueError(f"文件过小({len(data)}B)，可能无效")
            with open(dest, 'wb') as f:
                f.write(data)
            return len(data)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"    [FAIL] {url} -> {e}")
                return -1
    return -1


def main():
    total_ok, total_fail, total_bytes = 0, 0, 0
    for folder, items in MODELS.items():
        print(f"\n{'='*50}")
        print(f"  {folder} ({len(items)} 个模型)")
        print(f"{'='*50}")
        for src_name, url, save_name in items:
            dest = BASE_DIR / folder / save_name
            if dest.exists():
                print(f"  [SKIP] {save_name} (已存在)")
                total_ok += 1
                continue
            size = download_file(url, dest)
            if size > 0:
                print(f"  [OK] {save_name}  ({size/1024:.1f} KB)")
                total_ok += 1
                total_bytes += size
            else:
                total_fail += 1
            time.sleep(0.3)  # 礼貌延迟

    print(f"\n{'='*50}")
    print(f"  下载完成!")
    print(f"  成功: {total_ok}  失败: {total_fail}")
    print(f"  总大小: {total_bytes/1024/1024:.2f} MB")
    print(f"  保存路径: {BASE_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
