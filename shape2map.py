# -*- coding: utf-8 -*-
"""
棱镜模型 - 光路映射计算脚本
平行光从 z=+∞ 沿 -z 方向垂直照射 xOy 平面，
模型内部视为玻璃（折射率 n=1.43），追踪光线经过棱镜的折射路径，
计算入射平面 → 出射平面的 2D→2D 映射及其逆映射。

运行: python shape2map.py
可选参数:
  --resolution N    光线网格分辨率 (默认 128)
  --n_refract F     折射率 (默认 1.43)
  --shapes DIR      shapes 文件夹路径
  --maps DIR        输出 maps 文件夹路径
  --folder NAME     只处理指定子文件夹
  --model FILE      只处理指定模型文件
  --max_diverge F   光路最大横向发散距离 (默认 50.0, 超出停止追踪)
  --max_path F      光路最大累计路径长度 (默认 200.0)
  --skip_existing   跳过已存在 map 文件的模型 (不重复生成)

物理模型
  z = +∞  ← 平行光（方向 (0,0,-1)）
   ↓ ↓ ↓ ↓ ↓
   ┌─────────┐  ← 入射平面 (z = z_max + 1)
   │  空气 n=1│
   │ ┌─────┐ │  ← 玻璃棱镜 (n=1.43)
   │ │折射 │ │     入射面: Snell 折射进入
   │ │内部 │ │     出射面: Snell 折射出去
   │ └─────┘ │     (含全内反射处理)
   │  空气 n=1│
   └─────────┘  ← 出射记录平面 (z = -0.5)

NPZ 文件内部结构
data = np.load("maps/01_正多面体/正四面体_lightmap.npz")
data['input_grid']    # (res, res, 2) 入射光线 xOy 坐标
data['output_grid']   # (res, res, 2) 出射光线 xOy 坐标（正向映射）
data['valid_mask']    # (res, res) 有效光线 bool
data['inverse_map']   # (res, res, 2) 逆映射：出射坐标 → 入射坐标
data['inverse_valid'] # (res, res) 逆映射有效区域
data['resolution']    # 分辨率
data['n_refract']     # 折射率 1.43
data['bbox_min/max']  # 模型包围盒

# 完整运行（所有模型，128x128 分辨率）
python shape2map.py

# 高分辨率
python shape2map.py --resolution 256

# 只处理某个文件夹
python shape2map.py --folder "01_正多面体"

# 只处理单个模型
python shape2map.py --model "shapes/01_正多面体/正四面体.ply"

# 自定义折射率
python shape2map.py --n_refract 1.52

# 跳过已存在的 map, 限制发散距离
python shape2map.py --skip_existing --max_diverge 30

# 增量生成（跳过已有），限制发散
python shape2map.py --skip_existing --max_diverge 30

# 处理单个中文文件名模型
python shape2map.py --model "shapes/05_动物/兔子.ply"

# 严格发散限制（用于复杂多孔模型）
python shape2map.py --folder "04_多亏格曲面" --max_diverge 20 --max_path 100

--workers [n]
控制线程数
"""
import os, sys, math, argparse, time
import concurrent.futures
import numpy as np
from pathlib import Path

# ===================== 配置 =====================
DEFAULT_RESOLUTION = 128
DEFAULT_N_REFRACT = 1.43
MAX_BOUNCES = 6
EPSILON = 1e-7
OUTPUT_Z = -0.5  # 出射记录平面 (模型下方)
MAX_DIVERGE_DIST = 50.0   # 光路最大横向发散距离 (超出则停止追踪, 视为无效)
MAX_PATH_LENGTH = 200.0   # 光路最大累计路径长度 (超出则停止追踪)
MEMORY_BUDGET = 512 * 1024 * 1024  # 光线求交临时数组内存预算 (512MB), 用于自适应分块

# ===================== PLY 加载器 =====================
# PLY 属性类型 → numpy dtype 映射
_PLY_TYPE_MAP = {
    'char': 'i1', 'int8': 'i1',
    'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2',
    'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4',
    'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
}


def _parse_ply_header(header_lines):
    """解析 PLY 头部, 返回 (format, elements)"""
    fmt = 'ascii'
    elements = []
    current = None
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'format':
            fmt = parts[1]
        elif parts[0] == 'element':
            current = {'name': parts[1], 'count': int(parts[2]), 'props': []}
            elements.append(current)
        elif parts[0] == 'property' and current is not None:
            if parts[1] == 'list':
                current['props'].append({
                    'kind': 'list', 'count_type': parts[2],
                    'item_type': parts[3], 'name': parts[4],
                })
            else:
                current['props'].append({
                    'kind': 'scalar', 'type': parts[1], 'name': parts[2],
                })
    return fmt, elements


def _load_ply_ascii(f, vert_elem, face_elem, n_verts, n_faces):
    """解析 ASCII PLY (f 当前位置在 end_header 之后)"""
    rest = f.read().decode('utf-8', errors='ignore')
    lines = rest.splitlines()
    # 确定 x,y,z 属性索引
    xyz_idx = [0, 1, 2]
    if vert_elem:
        names = [p['name'] for p in vert_elem['props'] if p['kind'] == 'scalar']
        try:
            xyz_idx = [names.index('x'), names.index('y'), names.index('z')]
        except ValueError:
            xyz_idx = [0, 1, 2]
    vertices, faces = [], []
    vert_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if vert_count < n_verts:
            parts = line.split()
            vertices.append([float(parts[xyz_idx[0]]),
                             float(parts[xyz_idx[1]]),
                             float(parts[xyz_idx[2]])])
            vert_count += 1
        else:
            parts = line.split()
            if len(parts) >= 4:
                n = int(parts[0])
                idx = [int(parts[i + 1]) for i in range(n)]
                for i in range(1, n - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
            elif len(parts) == 3:
                faces.append([int(parts[0]), int(parts[1]), int(parts[2])])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def _load_ply_binary(f, vert_elem, face_elem, n_verts, n_faces, endian):
    """解析二进制 PLY (f 当前位置在 end_header 之后)"""
    vertices = np.zeros((n_verts, 3), dtype=np.float64)
    if vert_elem and n_verts > 0:
        dtype_list = []
        for p in vert_elem['props']:
            if p['kind'] != 'scalar':
                continue  # 顶点 list 属性罕见, 跳过
            dtype_list.append((p['name'], endian + _PLY_TYPE_MAP[p['type']]))
        dt = np.dtype(dtype_list)
        data = np.fromfile(f, dtype=dt, count=n_verts)
        for axis, key in enumerate(['x', 'y', 'z']):
            if key in data.dtype.names:
                vertices[:, axis] = data[key]

    faces = []
    if face_elem and n_faces > 0:
        list_prop = next((p for p in face_elem['props'] if p['kind'] == 'list'), None)
        if list_prop:
            ct = endian + _PLY_TYPE_MAP[list_prop['count_type']]
            it = endian + _PLY_TYPE_MAP[list_prop['item_type']]
            # list 属性之前的 scalar 字节数
            pre_bytes = 0
            for p in face_elem['props']:
                if p['kind'] == 'list':
                    break
                pre_bytes += np.dtype(endian + _PLY_TYPE_MAP[p['type']]).itemsize
            for _ in range(n_faces):
                if pre_bytes > 0:
                    f.read(pre_bytes)
                cnt = int(np.fromfile(f, dtype=ct, count=1)[0])
                idx = np.fromfile(f, dtype=it, count=cnt)
                for i in range(1, cnt - 1):
                    faces.append([int(idx[0]), int(idx[i]), int(idx[i + 1])])
    return vertices, np.array(faces, dtype=np.int32)


def load_ply(filepath):
    """加载 PLY 文件 (支持 ASCII / binary_little_endian / binary_big_endian)
    返回 vertices (N,3) 和 faces (M,3)"""
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        # 读取头部 (直到 end_header)
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                break
            try:
                text = line.decode('ascii').strip()
            except UnicodeDecodeError:
                text = line.decode('latin-1').strip()
            header_lines.append(text)
            if text == 'end_header':
                break

        fmt, elements = _parse_ply_header(header_lines)
        vert_elem = next((e for e in elements if e['name'] == 'vertex'), None)
        face_elem = next((e for e in elements if e['name'] == 'face'), None)
        n_verts = vert_elem['count'] if vert_elem else 0
        n_faces = face_elem['count'] if face_elem else 0

        if 'binary' in fmt:
            endian = '<' if 'little' in fmt else '>'
            return _load_ply_binary(f, vert_elem, face_elem, n_verts, n_faces, endian)
        else:
            return _load_ply_ascii(f, vert_elem, face_elem, n_verts, n_faces)


def load_obj(filepath):
    """加载 OBJ 文件"""
    vertices, faces = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                idx = []
                for p in parts[1:]:
                    idx.append(int(p.split('/')[0]) - 1)
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i+1]])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def load_model(filepath):
    """根据扩展名加载模型"""
    ext = Path(filepath).suffix.lower()
    if ext == '.ply':
        return load_ply(filepath)
    elif ext == '.obj':
        return load_obj(filepath)
    else:
        return None, None


# ===================== 光线追踪核心 =====================
def compute_face_normals(vertices, faces):
    """计算面法线（朝外）"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms
    # 确保法线朝外（相对于质心）
    center = vertices.mean(axis=0)
    face_centers = (v0 + v1 + v2) / 3.0
    outward = face_centers - center
    flip_mask = np.sum(normals * outward, axis=1) < 0
    normals[flip_mask] *= -1
    return normals


def ray_triangle_intersect(orig, dirn, v0, v1, v2):
    """Möller–Trumbore 光线-三角形求交（向量化）
    orig: (3,) or (N,3)
    dirn: (3,) or (N,3)
    v0, v1, v2: (M,3) 三角形顶点
    返回: t (距离), hit (bool), bary_u, bary_v
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    if dirn.ndim == 1:
        dirn = dirn[np.newaxis, :]
        orig = orig[np.newaxis, :]

    # 广播: (N,1,3) x (1,M,3) -> (N,M,3)
    N = orig.shape[0]
    M = v0.shape[0]

    d = dirn.reshape(N, 1, 3)
    o = orig.reshape(N, 1, 3)
    e1 = edge1.reshape(1, M, 3)
    e2 = edge2.reshape(1, M, 3)
    vv0 = v0.reshape(1, M, 3)

    h = np.cross(np.broadcast_to(d, (N, M, 3)), np.broadcast_to(e2, (N, M, 3)))
    a = np.sum(e1 * h, axis=2)  # (N, M)

    # 避免除零
    valid = np.abs(a) > EPSILON
    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]

    s = o - vv0
    u = f * np.sum(s * h, axis=2)

    q = np.cross(s, np.broadcast_to(e1, (N, M, 3)))
    v = f * np.sum(d * q, axis=2)

    t = f * np.sum(e2 * q, axis=2)

    hit = valid & (u >= -EPSILON) & (u <= 1.0 + EPSILON) & \
          (v >= -EPSILON) & (u + v <= 1.0 + EPSILON) & (t > EPSILON)

    return t, hit, u, v


def trace_rays_batch(origins, directions, vertices, faces, face_normals, n_refract,
                     bbox_center=None, max_diverge=MAX_DIVERGE_DIST, max_path=MAX_PATH_LENGTH,
                     chunk_size=None):
    """
    批量追踪光线穿过玻璃体。
    origins: (N, 3) 光线起点
    directions: (N, 3) 光线方向（归一化）
    bbox_center: (3,) 模型包围盒中心, 用于发散距离判定
    max_diverge: 光线偏离模型中心的最大横向距离, 超出则停止追踪并标记为无效
    max_path: 光线最大累计路径长度, 超出则停止追踪并标记为无效
    chunk_size: 每次求交的光线数 (None=根据面片数自适应, 避免内存爆炸)
    返回: exit_points (N,3), exit_dirs (N,3)
    """
    N = origins.shape[0]
    M = faces.shape[0]

    # 自适应分块: 控制 (chunk, M, 3) 临时数组内存不超过 MEMORY_BUDGET
    if chunk_size is None:
        chunk_size = max(1, int(MEMORY_BUDGET / max(M * 120, 1)))
        chunk_size = min(chunk_size, 4096)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cur_pos = origins.copy()
    cur_dir = directions.copy()
    inside = np.zeros(N, dtype=bool)
    active = np.ones(N, dtype=bool)
    path_len = np.zeros(N)  # 累计路径长度
    exit_points = np.full((N, 3), np.nan)
    exit_dirs = np.full((N, 3), np.nan)

    if bbox_center is None:
        bbox_center = vertices.mean(axis=0)

    n_air = 1.0
    n_glass = n_refract

    for bounce in range(MAX_BOUNCES):
        if not np.any(active):
            break

        # === 发散限制: 横向偏离模型中心太远的光线停止追踪 (标记为无效) ===
        horiz_dist = np.sqrt((cur_pos[:, 0] - bbox_center[0]) ** 2 +
                             (cur_pos[:, 1] - bbox_center[1]) ** 2)
        diverged = active & ((horiz_dist > max_diverge) | (path_len > max_path))
        if np.any(diverged):
            active[diverged] = False
            exit_points[diverged] = np.nan  # 确保发散光线不计入有效映射
            exit_dirs[diverged] = np.nan

        idx_active = np.where(active)[0]
        if len(idx_active) == 0:
            break

        # 分块处理活跃光线, 避免 (N_active × M) 求交数组内存爆炸
        for chunk_start in range(0, len(idx_active), chunk_size):
            chunk_idx = idx_active[chunk_start:chunk_start + chunk_size]
            o = cur_pos[chunk_idx]
            d = cur_dir[chunk_idx]
            Nc = len(chunk_idx)

            # 光线与所有三角形求交
            t_all, hit_all, u_all, v_all = ray_triangle_intersect(o, d, v0, v1, v2)

            # 对每条光线找最近交点
            t_all[~hit_all] = np.inf
            nearest_tri = np.argmin(t_all, axis=1)  # (Nc,)
            nearest_t = t_all[np.arange(Nc), nearest_tri]

            has_hit = nearest_t < np.inf

            # 更新无交点的光线：直接投射到输出平面
            no_hit_local = ~has_hit
            if np.any(no_hit_local):
                global_idx = chunk_idx[no_hit_local]
                # 投射到 OUTPUT_Z 平面
                t_plane = (OUTPUT_Z - cur_pos[global_idx, 2]) / cur_dir[global_idx, 2]
                valid_plane = t_plane > 0
                if np.any(valid_plane):
                    gi_valid = global_idx[valid_plane]
                    exit_points[gi_valid] = cur_pos[gi_valid] + cur_dir[gi_valid] * t_plane[valid_plane, np.newaxis]
                    exit_dirs[gi_valid] = cur_dir[gi_valid]
                active[global_idx] = False

            # 有交点的光线：折射
            hit_local = has_hit
            if not np.any(hit_local):
                continue

            hit_global = chunk_idx[hit_local]
            hit_tri = nearest_tri[hit_local]
            hit_t = nearest_t[hit_local]

            # 交点位置
            hit_pos = cur_pos[hit_global] + cur_dir[hit_global] * hit_t[:, np.newaxis]

            # 累计路径长度 (用于发散限制判定)
            path_len[hit_global] += hit_t

            # 交点处法线
            hit_normals = face_normals[hit_tri]

            # 确保法线面向入射方向（如果光线从外面进来，法线应该与光线方向相反）
            dot_nd = np.sum(hit_normals * cur_dir[hit_global], axis=1)
            # 如果 dot > 0，法线与光线同向，说明光线从内部打到表面（出射）
            entering = dot_nd < 0  # 从外部进入

            # 折射计算 (Snell's law vector form) —— 全向量化, 无数组逐元素 Python 循环
            d_in = cur_dir[hit_global]                       # (Nh, 3) 入射方向
            n_surf = hit_normals.copy()                      # (Nh, 3) 表面法线
            # 折射率比 eta: 进入=air/glass, 出射=glass/air; 出射时法线翻转
            eta = np.where(entering, n_air / n_glass, n_glass / n_air)  # (Nh,)
            n_surf = np.where(entering[:, None], n_surf, -n_surf)
            cos_i = np.clip(-np.sum(d_in * n_surf, axis=1), 0.0, 1.0)   # (Nh,)
            sin2_t = eta * eta * (1.0 - cos_i * cos_i)       # (Nh,)
            tir = sin2_t > 1.0                               # 全内反射掩码

            # --- 全内反射: d' = d + 2cos_i·n ---
            if np.any(tir):
                d_reflect = d_in[tir] + 2.0 * cos_i[tir, None] * n_surf[tir]
                d_reflect /= (np.linalg.norm(d_reflect, axis=1, keepdims=True) + 1e-12)
                gi_tir = hit_global[tir]
                cur_dir[gi_tir] = d_reflect
                cur_pos[gi_tir] = hit_pos[tir] + d_reflect * (EPSILON * 10)

            # --- 折射: d' = eta·d + (eta·cos_i - cos_t)·n ---
            refr = ~tir
            if np.any(refr):
                cos_t = np.sqrt(np.clip(1.0 - sin2_t[refr], 0.0, None))
                eta_r = eta[refr]
                d_refract = eta_r[:, None] * d_in[refr] + \
                            (eta_r * cos_i[refr] - cos_t)[:, None] * n_surf[refr]
                d_refract /= (np.linalg.norm(d_refract, axis=1, keepdims=True) + 1e-12)
                gi_refr = hit_global[refr]
                cur_dir[gi_refr] = d_refract
                cur_pos[gi_refr] = hit_pos[refr] + d_refract * (EPSILON * 10)
                # 出射 (非进入) → 记录并停止追踪
                exiting_local = (~entering)[refr]
                if np.any(exiting_local):
                    gi_exit = gi_refr[exiting_local]
                    exit_points[gi_exit] = hit_pos[refr][exiting_local]
                    exit_dirs[gi_exit] = d_refract[exiting_local]
                    active[gi_exit] = False
                # 进入的光线继续追踪（inside）

    # 对仍然 active 的光线，投射到输出平面
    still_active = np.where(active)[0]
    if len(still_active) > 0:
        for gi in still_active:
            if abs(cur_dir[gi, 2]) > 1e-10:
                t_plane = (OUTPUT_Z - cur_pos[gi, 2]) / cur_dir[gi, 2]
                if t_plane > 0:
                    exit_points[gi] = cur_pos[gi] + cur_dir[gi] * t_plane
                    exit_dirs[gi] = cur_dir[gi]

    return exit_points, exit_dirs


# ===================== 映射计算 =====================
def compute_mapping(vertices, faces, resolution, n_refract,
                    max_diverge=MAX_DIVERGE_DIST, max_path=MAX_PATH_LENGTH):
    """
    计算光路映射。
    返回:
      input_grid: (res, res, 2) 入射光线 xOy 坐标
      output_grid: (res, res, 2) 出射光线在 OUTPUT_Z 平面的 xOy 坐标
      valid_mask: (res, res) 有效光线标记
    """
    # 计算模型包围盒
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    bbox_center = (vmin + vmax) / 2.0
    margin = 0.15 * max(vmax[0] - vmin[0], vmax[1] - vmin[1])

    x_min, x_max = vmin[0] - margin, vmax[0] + margin
    y_min, y_max = vmin[1] - margin, vmax[1] + margin

    # 生成光线网格
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)

    input_grid = np.stack([xx, yy], axis=-1)  # (res, res, 2)

    # 光线起点：模型上方
    z_start = vmax[2] + 1.0
    origins = np.stack([xx.ravel(), yy.ravel(), np.full(resolution * resolution, z_start)], axis=-1)
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))

    # 计算面法线
    face_normals = compute_face_normals(vertices, faces)

    # 追踪光线 (带发散限制)
    exit_pts, exit_dirs = trace_rays_batch(
        origins, directions, vertices, faces, face_normals, n_refract,
        bbox_center=bbox_center, max_diverge=max_diverge, max_path=max_path
    )

    # 出射光线投射到 OUTPUT_Z 平面
    output_grid = np.full((resolution * resolution, 2), np.nan)
    valid = ~np.isnan(exit_pts[:, 0])

    # 对于 exit_points 已经在 OUTPUT_Z 的，直接取 xy
    # 对于还在模型表面的出射点，沿 exit_dir 投射到 OUTPUT_Z
    for i in range(len(exit_pts)):
        if not valid[i]:
            continue
        if abs(exit_pts[i, 2] - OUTPUT_Z) < 0.01:
            output_grid[i] = exit_pts[i, :2]
        elif not np.isnan(exit_dirs[i, 2]) and abs(exit_dirs[i, 2]) > 1e-10:
            t = (OUTPUT_Z - exit_pts[i, 2]) / exit_dirs[i, 2]
            if t > 0:
                hit = exit_pts[i] + exit_dirs[i] * t
                output_grid[i] = hit[:2]
            else:
                output_grid[i] = exit_pts[i, :2]
        else:
            output_grid[i] = exit_pts[i, :2]

    output_grid = output_grid.reshape(resolution, resolution, 2)
    valid_mask = valid.reshape(resolution, resolution)

    return input_grid, output_grid, valid_mask


def compute_inverse_mapping(input_grid, output_grid, valid_mask, resolution):
    """
    计算逆映射：从出射平面坐标 → 入射平面坐标
    使用最近邻插值在散乱数据上构建逆映射
    """
    # 有效的 (output_xy -> input_xy) 对
    out_pts = output_grid[valid_mask]   # (K, 2)
    in_pts = input_grid[valid_mask]     # (K, 2)

    if len(out_pts) == 0:
        return np.full((resolution, resolution, 2), np.nan), np.zeros((resolution, resolution), dtype=bool)

    # 在出射平面上生成规则网格
    out_x_min, out_x_max = np.nanmin(out_pts[:, 0]), np.nanmax(out_pts[:, 0])
    out_y_min, out_y_max = np.nanmin(out_pts[:, 1]), np.nanmax(out_pts[:, 1])

    # 加一点边距
    pad_x = (out_x_max - out_x_min) * 0.05 + 1e-6
    pad_y = (out_y_max - out_y_min) * 0.05 + 1e-6
    out_x_min -= pad_x; out_x_max += pad_x
    out_y_min -= pad_y; out_y_max += pad_y

    inv_xs = np.linspace(out_x_min, out_x_max, resolution)
    inv_ys = np.linspace(out_y_min, out_y_max, resolution)
    inv_xx, inv_yy = np.meshgrid(inv_xs, inv_ys)
    inv_query = np.stack([inv_xx.ravel(), inv_yy.ravel()], axis=-1)  # (res*res, 2)

    # 最近邻查找（KD-tree）
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(out_pts)
        dists, indices = tree.query(inv_query, k=1)
        # 距离过远的视为无效
        max_dist = max((out_x_max - out_x_min), (out_y_max - out_y_min)) / resolution * 2.0
        inv_valid = dists < max_dist
        inv_map = np.full((resolution * resolution, 2), np.nan)
        inv_map[inv_valid] = in_pts[indices[inv_valid]]
    except ImportError:
        # fallback: 暴力搜索（慢）
        inv_map = np.full((resolution * resolution, 2), np.nan)
        inv_valid = np.zeros(resolution * resolution, dtype=bool)
        for i in range(len(inv_query)):
            d = np.sum((out_pts - inv_query[i]) ** 2, axis=1)
            idx = np.argmin(d)
            if d[idx] < (max(out_x_max - out_x_min, out_y_max - out_y_min) / resolution * 2) ** 2:
                inv_map[i] = in_pts[idx]
                inv_valid[i] = True

    inv_map = inv_map.reshape(resolution, resolution, 2)
    inv_valid = inv_valid.reshape(resolution, resolution)
    return inv_map, inv_valid


# ===================== 文件保存 =====================
def save_mapping(filepath, input_grid, output_grid, valid_mask, inv_map, inv_valid, meta):
    """保存映射为 .npz 文件（含元数据）"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filepath,
        input_grid=input_grid,       # (res, res, 2) 入射坐标
        output_grid=output_grid,     # (res, res, 2) 出射坐标
        valid_mask=valid_mask,       # (res, res) bool
        inverse_map=inv_map,         # (res, res, 2) 逆映射
        inverse_valid=inv_valid,     # (res, res) bool
        resolution=np.array(meta['resolution']),
        n_refract=np.array(meta['n_refract']),
        output_z=np.array(meta['output_z']),
        bbox_min=np.array(meta['bbox_min']),
        bbox_max=np.array(meta['bbox_max']),
    )


def save_mapping_txt(filepath, input_grid, output_grid, valid_mask):
    """另存一份可读的 TXT 映射表（仅有效点）"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 光路映射: input_x input_y -> output_x output_y\n")
        f.write(f"# 有效光线数: {np.sum(valid_mask)}\n")
        f.write("# columns: in_x in_y out_x out_y\n")
        res = input_grid.shape[0]
        for i in range(res):
            for j in range(res):
                if valid_mask[i, j]:
                    ix, iy = input_grid[i, j]
                    ox, oy = output_grid[i, j]
                    f.write(f"{ix:.6f} {iy:.6f} {ox:.6f} {oy:.6f}\n")


# ===================== 主流程 =====================
def normalize_model(vertices, target_size=2.0):
    """归一化模型: X-Y 居中, z_min 置于 0, 最大跨度缩放到 target_size。
    使光线追踪尺度无关——不同原始尺度的模型归一化后一致追踪,
    且归一化位移场模式与原始一致(训练时位移按 disp_max 归一化, 均匀缩放不变)。"""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = (vmax - vmin).max()
    if extent > 1e-9:
        vertices = vertices * (target_size / extent)
    # X-Y 居中, z 置于 0 (正立于 XOY 平面)
    vertices[:, 0] -= (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
    vertices[:, 1] -= (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
    vertices[:, 2] -= vertices[:, 2].min()
    return vertices


def process_model(model_path, output_dir, resolution, n_refract,
                  max_diverge=MAX_DIVERGE_DIST, max_path=MAX_PATH_LENGTH):
    """处理单个模型"""
    vertices, faces = load_model(model_path)
    if vertices is None or len(vertices) == 0 or len(faces) == 0:
        return False, "无法加载模型"

    # 归一化: X-Y居中, z置于0, 最大跨度缩放到标准尺寸 (尺度无关, 修复大尺度模型发散误判)
    vertices = normalize_model(vertices)

    input_grid, output_grid, valid_mask = compute_mapping(
        vertices, faces, resolution, n_refract,
        max_diverge=max_diverge, max_path=max_path
    )

    inv_map, inv_valid = compute_inverse_mapping(
        input_grid, output_grid, valid_mask, resolution
    )

    meta = {
        'resolution': resolution,
        'n_refract': n_refract,
        'output_z': OUTPUT_Z,
        'bbox_min': vertices.min(axis=0).tolist(),
        'bbox_max': vertices.max(axis=0).tolist(),
    }

    stem = Path(model_path).stem
    npz_path = output_dir / f"{stem}_lightmap.npz"
    txt_path = output_dir / f"{stem}_lightmap.txt"

    save_mapping(npz_path, input_grid, output_grid, valid_mask, inv_map, inv_valid, meta)
    save_mapping_txt(txt_path, input_grid, output_grid, valid_mask)

    n_valid = int(np.sum(valid_mask))
    return True, f"{n_valid}/{resolution*resolution} 条光线有效"


# ===================== 并行工作函数 =====================
def _worker_task(task):
    """进程池工作函数: 处理单个模型 (顶层函数以便 pickle 序列化)。
    返回 (模型路径str, 是否成功, 消息, 错误信息)。各模型独立, 进程间无共享状态。"""
    model_path, out_dir, resolution, n_refract, max_diverge, max_path = task
    try:
        success, msg = process_model(
            model_path, out_dir, resolution, n_refract,
            max_diverge=max_diverge, max_path=max_path
        )
        return (str(model_path), success, msg, None)
    except Exception as e:
        return (str(model_path), False, None, f"{type(e).__name__}: {e}")


def main():
    # Windows 控制台中文输出支持
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

    parser = argparse.ArgumentParser(description="棱镜光路映射计算")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="光线网格分辨率")
    parser.add_argument("--n_refract", type=float, default=DEFAULT_N_REFRACT, help="折射率")
    parser.add_argument("--shapes", type=str, default=None, help="shapes 文件夹路径")
    parser.add_argument("--maps", type=str, default=None, help="输出 maps 文件夹路径")
    parser.add_argument("--folder", type=str, default=None, help="只处理指定子文件夹名")
    parser.add_argument("--model", type=str, default=None, help="只处理指定模型文件")
    parser.add_argument("--max_diverge", type=float, default=MAX_DIVERGE_DIST,
                        help="光路最大横向发散距离 (超出停止追踪)")
    parser.add_argument("--max_path", type=float, default=MAX_PATH_LENGTH,
                        help="光路最大累计路径长度 (超出停止追踪)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在 map 文件的模型 (不重复生成)")
    parser.add_argument("--workers", type=int, default=0,
                        help="并行进程数 (0=自动=CPU核心数, 1=串行调试)")
    args = parser.parse_args()

    # 确定并行进程数
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    base_dir = Path(__file__).parent
    shapes_dir = Path(args.shapes) if args.shapes else base_dir / "shapes"
    maps_dir = Path(args.maps) if args.maps else base_dir / "maps"

    print(f"棱镜光路映射计算器")
    print(f"  分辨率: {args.resolution} x {args.resolution}")
    print(f"  折射率: {args.n_refract}")
    print(f"  发散限制: 横向 {args.max_diverge} / 路径 {args.max_path}")
    print(f"  并行进程: {workers}")
    print(f"  跳过已存在: {'是' if args.skip_existing else '否'}")
    print(f"  模型目录: {shapes_dir}")
    print(f"  输出目录: {maps_dir}")
    print(f"{'='*60}")

    # 收集模型文件
    model_files = []
    if args.model:
        mp = Path(args.model)
        if not mp.is_absolute():
            mp = base_dir / mp
        model_files = [(mp.resolve(), maps_dir)]
    else:
        for sub in sorted(shapes_dir.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name.startswith('_'):  # 跳过 _cache 等辅助目录
                continue
            if args.folder and sub.name != args.folder:
                continue
            out_sub = maps_dir / sub.name
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in ('.ply', '.obj'):
                    model_files.append((f, out_sub))

    total = len(model_files)
    ok_count = 0
    skip_count = 0
    t0 = time.time()

    def _rel(p):
        try:
            return p.relative_to(shapes_dir)
        except ValueError:
            return p.name

    # 预过滤: skip_existing 在提交前剔除已完成模型, 构建任务列表
    tasks = []
    for model_path, out_dir in model_files:
        if args.skip_existing:
            npz_path = out_dir / f"{model_path.stem}_lightmap.npz"
            if npz_path.exists():
                print(f"  [SKIP] {_rel(model_path)} (已存在)")
                skip_count += 1
                continue
        tasks.append((model_path, out_dir, args.resolution, args.n_refract,
                      args.max_diverge, args.max_path))

    n_task = len(tasks)
    rel_map = {str(t[0]): _rel(t[0]) for t in tasks}  # 路径->相对名, 供并行结果输出

    if n_task == 0:
        print("  无待处理模型")
    elif workers == 1 or n_task == 1:
        # 串行处理 (单进程, 便于调试)
        for idx, task in enumerate(tasks):
            rel_name = rel_map[str(task[0])]
            print(f"  [{idx+1}/{n_task}] {rel_name} ...", end=" ", flush=True)
            _, success, msg, err = _worker_task(task)
            if err:
                print(f"ERROR ({err})")
            elif success:
                print(f"OK ({msg})"); ok_count += 1
            else:
                print(f"SKIP ({msg})")
    else:
        # 多进程并行: 限制每进程 BLAS 线程数, 避免线程过度订阅 (spawn 子进程继承环境变量)
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[var] = '1'
        print(f"  多进程并行处理 {n_task} 个模型 (workers={workers}) ...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker_task, t): t for t in tasks}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                model_path_str, success, msg, err = future.result()
                done += 1
                rel_name = rel_map.get(model_path_str, Path(model_path_str).name)
                if err:
                    print(f"  [{done}/{n_task}] {rel_name} ... ERROR ({err})")
                elif success:
                    print(f"  [{done}/{n_task}] {rel_name} ... OK ({msg})")
                    ok_count += 1
                else:
                    print(f"  [{done}/{n_task}] {rel_name} ... SKIP ({msg})")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成! 成功 {ok_count}/{total}, 跳过 {skip_count}, 耗时 {elapsed:.1f}s")
    print(f"  输出: {maps_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()