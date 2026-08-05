"""
棱镜模型 - PLY 模型文件批量生成器
所有模型正立于 XOY 平面之上（Z轴朝上，z>=0）
运行: python generate_shapes.py
"""
import os, math, sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent / "shapes"

# ===================== PLY 写入 =====================
def weld_vertices(verts, faces, tol=1e-3):
    """合并空间上接近的顶点(容差 tol)。
    双环面 / 拼接物体的关键:让两个独立网格在拼接处共享顶点,消除视觉裂缝。"""
    from collections import defaultdict
    bucket = defaultdict(list)
    for i, v in enumerate(verts):
        key = (int(v[0] / tol), int(v[1] / tol), int(v[2] / tol))
        bucket[key].append(i)
    rep = list(range(len(verts)))  # rep[i] = 代表顶点
    for idx_list in bucket.values():
        for idx in idx_list[1:]:
            rep[idx] = idx_list[0]
    # 重映射到紧凑索引
    new_verts = []
    remap = {}
    for i, v in enumerate(verts):
        r = rep[i]
        if r not in remap:
            remap[r] = len(new_verts)
            new_verts.append(v)
    new_faces = []
    for f in faces:
        nf = tuple(remap[rep[idx]] for idx in f)
        # 退化检查:不能 3 个相同顶点
        if len(set(nf)) == 3:
            new_faces.append(nf)
    return new_verts, new_faces

def write_ply(filepath, vertices, faces):
    """写入带顶点法线的 ASCII PLY（三角面片）"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    verts = np.array(vertices, dtype=np.float64)
    tris = np.array(faces, dtype=np.int32)
    # 计算面法线
    v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(fn, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    fn = fn / norm
    # 顶点法线
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, tris[:, i], fn)
    vn_norm = np.linalg.norm(vn, axis=1, keepdims=True)
    vn_norm[vn_norm < 1e-12] = 1.0
    vn = vn / vn_norm
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write(f"element face {len(tris)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(len(verts)):
            f.write(f"{verts[i,0]:.6f} {verts[i,1]:.6f} {verts[i,2]:.6f} "
                    f"{vn[i,0]:.6f} {vn[i,1]:.6f} {vn[i,2]:.6f}\n")
        for tri in tris:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

def ensure_upright(verts):
    """确保模型底部在 z=0"""
    verts = np.array(verts, dtype=np.float64)
    verts[:, 2] -= verts[:, 2].min()
    return verts.tolist()

def add_ellipsoid(verts, faces, cx, cy, cz, rx, ry, rz, rings=10, sectors=12):
    """向 verts/faces 添加椭球"""
    start = len(verts)
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors):
            theta = 2 * math.pi * s / sectors
            verts.append((cx + rx*math.sin(phi)*math.cos(theta),
                          cy + ry*math.sin(phi)*math.sin(theta),
                          cz + rz*math.cos(phi)))
    for r in range(rings):
        for s in range(sectors):
            c = start + r*sectors + s
            n = start + r*sectors + (s+1)%sectors
            cn = start + (r+1)*sectors + s
            nn = start + (r+1)*sectors + (s+1)%sectors
            faces.append((c, n, cn)); faces.append((n, nn, cn))

def add_cylinder(verts, faces, cx, cy, z0, z1, r, seg=12, cap=True):
    """添加圆柱"""
    start = len(verts)
    for i in range(seg):
        t = 2*math.pi*i/seg
        verts.append((cx+r*math.cos(t), cy+r*math.sin(t), z0))
    for i in range(seg):
        t = 2*math.pi*i/seg
        verts.append((cx+r*math.cos(t), cy+r*math.sin(t), z1))
    for i in range(seg):
        ni = (i+1)%seg
        faces.append((start+i, start+ni, start+seg+i))
        faces.append((start+ni, start+seg+ni, start+seg+i))
    if cap:
        bc = len(verts); verts.append((cx, cy, z0))
        tc = len(verts); verts.append((cx, cy, z1))
        for i in range(seg):
            ni = (i+1)%seg
            faces.append((bc, start+ni, start+i))
            faces.append((tc, start+seg+i, start+seg+ni))

def add_cone(verts, faces, cx, cy, z0, z1, r, seg=12):
    """添加圆锥（底z0，顶z1）"""
    start = len(verts)
    verts.append((cx, cy, z1))  # apex
    for i in range(seg):
        t = 2*math.pi*i/seg
        verts.append((cx+r*math.cos(t), cy+r*math.sin(t), z0))
    bc = len(verts); verts.append((cx, cy, z0))
    for i in range(seg):
        ni = (i+1)%seg
        faces.append((start, start+1+i, start+1+ni))
        faces.append((bc, start+1+ni, start+1+i))

def add_box(verts, faces, cx, cy, cz, hx, hy, hz):
    """添加长方体，中心(cx,cy,cz)，半尺寸(hx,hy,hz)"""
    b = len(verts)
    for dz in [-1, 1]:
        for dy in [-1, 1]:
            for dx in [-1, 1]:
                verts.append((cx+dx*hx, cy+dy*hy, cz+dz*hz))
    # 顶点顺序: 0(-,-,-) 1(-,-,+) 2(-,+,-) 3(-,+,+) 4(+,-,-) 5(+,-,+) 6(+,+,-) 7(+,+,+)
    quads = [(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    for q in quads:
        faces.append((b+q[0], b+q[1], b+q[2]))
        faces.append((b+q[0], b+q[2], b+q[3]))

def add_tube(verts, faces, path_pts, radius, seg=8):
    """沿路径生成管状网格"""
    start = len(verts)
    n = len(path_pts)
    for i in range(n):
        p = np.array(path_pts[i], dtype=float)
        if i < n-1:
            tangent = np.array(path_pts[i+1]) - p
        else:
            tangent = p - np.array(path_pts[i-1])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        # 构造正交基
        up = np.array([0,0,1.0])
        if abs(np.dot(tangent, up)) > 0.99:
            up = np.array([1.0, 0, 0])
        right = np.cross(tangent, up); right /= np.linalg.norm(right)+1e-12
        up2 = np.cross(right, tangent)
        for j in range(seg):
            theta = 2*math.pi*j/seg
            offset = radius*(math.cos(theta)*right + math.sin(theta)*up2)
            verts.append(tuple(p + offset))
    for i in range(n-1):
        for j in range(seg):
            nj = (j+1)%seg
            c = start + i*seg + j
            faces.append((c, c+seg, start+i*seg+nj))
            faces.append((start+i*seg+nj, c+seg, start+(i+1)*seg+nj))

# ===================== 基础几何体 =====================
def gen_tetrahedron():
    a = 1.0
    v = [(a,a,a),(a,-a,-a),(-a,a,-a),(-a,-a,a)]
    f = [(0,1,2),(0,3,1),(0,2,3),(1,3,2)]
    return ensure_upright(v), f

def gen_cube():
    v, f = [], []
    add_box(v, f, 0, 0, 1, 1, 1, 1)
    return v, f

def gen_octahedron():
    a = 1.0
    v = [(a,0,0),(-a,0,0),(0,a,0),(0,-a,0),(0,0,a),(0,0,-a)]
    f = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    return ensure_upright(v), f

def gen_dodecahedron():
    phi = (1+math.sqrt(5))/2; ip = 1/phi
    v = [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1),
         (0,ip,phi),(0,ip,-phi),(0,-ip,phi),(0,-ip,-phi),
         (ip,phi,0),(ip,-phi,0),(-ip,phi,0),(-ip,-phi,0),
         (phi,0,ip),(phi,0,-ip),(-phi,0,ip),(-phi,0,-ip)]
    pents = [(0,8,10,2,16),(0,16,17,1,12),(0,12,14,4,8),(1,17,3,11,9),(1,9,5,14,12),
             (2,10,6,15,13),(2,13,3,17,16),(3,13,15,7,11),(4,14,5,19,18),
             (4,18,6,10,8),(5,9,11,7,19),(6,18,19,7,15)]
    f = []
    for p in pents:
        for i in range(1, len(p)-1):
            f.append((p[0], p[i], p[i+1]))
    return ensure_upright(v), f

def gen_icosahedron():
    phi = (1+math.sqrt(5))/2
    v = [(-1,phi,0),(1,phi,0),(-1,-phi,0),(1,-phi,0),(0,-1,phi),(0,1,phi),
         (0,-1,-phi),(0,1,-phi),(phi,0,-1),(phi,0,1),(-phi,0,-1),(-phi,0,1)]
    f = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),
         (10,7,6),(7,1,8),(3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
         (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    return ensure_upright(v), f

def _convex_hull(pts):
    """用 scipy 凸包三角化"""
    from scipy.spatial import ConvexHull
    pts = np.array(pts, dtype=np.float64)
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    idx_map = {old: new for new, old in enumerate(hull.vertices)}
    faces = []
    center = verts.mean(axis=0)
    for s in hull.simplices:
        tri = (idx_map[s[0]], idx_map[s[1]], idx_map[s[2]])
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1-v0, v2-v0)
        if np.dot(n, v0-center) < 0:
            tri = (tri[0], tri[2], tri[1])
        faces.append(tri)
    return ensure_upright(verts.tolist()), faces

def gen_truncated_tetrahedron():
    coords = []
    for s in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]:
        coords += [(3*s[0],s[1],s[2]),(s[0],3*s[1],s[2]),(s[0],s[1],3*s[2])]
    return _convex_hull([(c[0]/3,c[1]/3,c[2]/3) for c in coords])

def gen_truncated_cube():
    xi = math.sqrt(2)-1; v = []
    for sx in [-1,1]:
        for sy in [-1,1]:
            for sz in [-1,1]:
                v += [(sx,sy,sz*xi),(sx,sy*xi,sz),(sx*xi,sy,sz)]
    return _convex_hull(v)

def gen_cuboctahedron():
    v = []
    for sx in [-1,1]:
        for sy in [-1,1]:
            v += [(sx,sy,0),(sx,0,sy),(0,sx,sy)]
    return _convex_hull(v)

def gen_icosidodecahedron():
    phi = (1+math.sqrt(5))/2
    v = [(0,0,phi),(0,0,-phi),(0,phi,0),(0,-phi,0),(phi,0,0),(-phi,0,0),
         (0.5,phi/2,(1+phi)/2),(0.5,phi/2,-(1+phi)/2),(-0.5,phi/2,(1+phi)/2),
         (-0.5,phi/2,-(1+phi)/2),(0.5,-phi/2,(1+phi)/2),(0.5,-phi/2,-(1+phi)/2),
         (-0.5,-phi/2,(1+phi)/2),(-0.5,-phi/2,-(1+phi)/2),
         ((1+phi)/2,0.5,phi/2),((1+phi)/2,-0.5,phi/2),((1+phi)/2,0.5,-phi/2),
         ((1+phi)/2,-0.5,-phi/2),(-(1+phi)/2,0.5,phi/2),(-(1+phi)/2,-0.5,phi/2),
         (-(1+phi)/2,0.5,-phi/2),(-(1+phi)/2,-0.5,-phi/2),
         (phi/2,(1+phi)/2,0.5),(phi/2,(1+phi)/2,-0.5),(-phi/2,(1+phi)/2,0.5),
         (-phi/2,(1+phi)/2,-0.5),(phi/2,-(1+phi)/2,0.5),(phi/2,-(1+phi)/2,-0.5),
         (-phi/2,-(1+phi)/2,0.5),(-phi/2,-(1+phi)/2,-0.5)]
    return _convex_hull(v)

def gen_rhombicuboctahedron():
    xi = 1+math.sqrt(2); v = []
    for sx in [-1,1]:
        for sy in [-1,1]:
            for sz in [-1,1]:
                v += [(sx,sy,sz*xi),(sx,sy*xi,sz),(sx*xi,sy,sz)]
    return _convex_hull(v)

def gen_sphere(rings=20, sectors=28):
    v, f = [], []
    R = 1.0
    for r in range(rings+1):
        phi = math.pi*r/rings
        for s in range(sectors):
            theta = 2*math.pi*s/sectors
            v.append((R*math.sin(phi)*math.cos(theta), R*math.sin(phi)*math.sin(theta), R*math.cos(phi)+R))
    for r in range(rings):
        for s in range(sectors):
            c = r*sectors+s; n = r*sectors+(s+1)%sectors
            cn = (r+1)*sectors+s; nn = (r+1)*sectors+(s+1)%sectors
            f.append((c,n,cn)); f.append((n,nn,cn))
    return v, f

def gen_cone_shape(seg=24):
    v, f = [], []
    add_cone(v, f, 0, 0, 0, 2.0, 1.0, seg)
    return v, f

def gen_pyramid_3():
    v, f = [], []
    add_cone(v, f, 0, 0, 0, 1.8, 1.0, 3)
    return v, f

def gen_pyramid_4():
    v, f = [], []
    add_cone(v, f, 0, 0, 0, 1.8, 1.0, 4)
    return v, f

def gen_pyramid_6():
    v, f = [], []
    add_cone(v, f, 0, 0, 0, 1.8, 1.0, 6)
    return v, f

def gen_frustum():
    """圆台"""
    v, f = [], []
    seg = 20; r1, r2, h = 1.0, 0.5, 1.5
    for i in range(seg):
        t = 2*math.pi*i/seg
        v.append((r1*math.cos(t), r1*math.sin(t), 0))
    for i in range(seg):
        t = 2*math.pi*i/seg
        v.append((r2*math.cos(t), r2*math.sin(t), h))
    bc = len(v); v.append((0,0,0))
    tc = len(v); v.append((0,0,h))
    for i in range(seg):
        ni = (i+1)%seg
        f.append((i, ni, seg+i)); f.append((ni, seg+ni, seg+i))
        f.append((bc, ni, i)); f.append((tc, seg+i, seg+ni))
    return v, f

def gen_torus(R=1.0, r=0.4, rings=28, sides=18):
    v, f = [], []
    for i in range(rings):
        theta = 2*math.pi*i/rings
        for j in range(sides):
            phi = 2*math.pi*j/sides
            v.append(((R+r*math.cos(phi))*math.cos(theta),
                      (R+r*math.cos(phi))*math.sin(theta),
                      r*math.sin(phi)+R+r))
    for i in range(rings):
        for j in range(sides):
            c = i*sides+j; nj = i*sides+(j+1)%sides
            ni = ((i+1)%rings)*sides+j; nij = ((i+1)%rings)*sides+(j+1)%sides
            f.append((c,nj,ni)); f.append((nj,nij,ni))
    return v, f

def gen_double_torus():
    """双环面(亏格 2):两个 R=0.7, r=0.3 的 torus + 沿 x 方向的连接桥。
    桥的端点直接复用第一个 torus 的 θ=0 切片(已在 v1)和第二个 torus 的 θ=0 切片(已在 v2),
    共享 16 个端点 → 拓扑上完全闭合,无独立小圆环。"""
    R, r = 0.7, 0.3
    rings, sides = 28, 16
    gap = 0.2

    # 两个完整 torus(各自封闭)
    v1, f1 = gen_torus(R, r, rings, sides)
    cx2 = 2.0 * (R + r) + gap     # = 1.4 + 0.2 = 1.6
    v2, f2 = gen_torus(R, r, rings, sides)
    v2 = [(x + cx2, y, z) for x, y, z in v2]
    f2 = [(a + len(v1), b + len(v1), c + len(v1)) for a, b, c in f2]

    # 桥的端点 = v1 的 θ=0 切片 (i=0, j=0..15) 16 个点 + v2 的 θ=0 切片 16 个点
    # 中间 n_seg-1 个新切片,沿 x 线性插值(端面 y/z 完全相同,只 x 不同)
    n_seg = 10  # 桥的中间段数
    bv = []
    # 中间切片(k=1..n_seg-1)
    for k in range(1, n_seg):
        t = k / n_seg
        for j in range(sides):
            phi = 2 * math.pi * j / sides
            # 端面 y=0, z=R+r+r*sin(phi)
            x1 = R + r * math.cos(phi)        # v1[i=0,j].x
            x2 = cx2 + R + r * math.cos(phi)  # v2[i=0,j].x
            x  = (1.0 - t) * x1 + t * x2
            y  = 0.0
            z  = R + r + r * math.sin(phi)
            bv.append((x, y, z))

    # 桥的面:端面 0 (v1) → 中间 n_seg-1 切片 → 端面 n_seg (v2)
    bf = []
    base_bv  = len(v1) + len(v2)
    base_v2   = len(v1)
    # 每个 j 走 16 段,n_seg+1 个切片
    for j in range(sides):
        jn = (j + 1) % sides
        # 切片 k 的 j 点索引
        def idx(k, j_):
            if k == 0:        return 0 * sides + j_          # v1 端
            if k == n_seg:   return base_v2 + 0 * sides + j_  # v2 端
            return base_bv + (k - 1) * sides + j_           # 中间
        for k in range(n_seg):
            v_a = idx(k,     j)
            v_b = idx(k + 1, j)
            v_c = idx(k,     jn)
            v_d = idx(k + 1, jn)
            bf.append((v_a, v_b, v_d))
            bf.append((v_a, v_d, v_c))

    return v1 + v2 + bv, f1 + f2 + bf

def gen_klein_bottle(su=48, sv=24):
    """克莱因瓶 (figure-8 immersion, v 方向闭合 + 加密以减少自交处视觉断裂)"""
    v, f = [], []
    a = 2.0  # 瓶身半径(>= 2 让瓶颈不与瓶身过度相交)
    for i in range(su):
        u = 2 * math.pi * i / su
        cu  = math.cos(u)
        su2 = math.sin(u)
        cu2 = math.cos(u / 2)
        su_2 = math.sin(u / 2)
        for j in range(sv):
            vv = 2 * math.pi * j / sv
            svv  = math.sin(vv)
            s2v  = math.sin(2 * vv)
            r   = a + cu2 * svv - su_2 * s2v
            x   = r * cu
            y   = r * su2
            z   = su_2 * svv + cu2 * s2v
            v.append((x * 0.5, y * 0.5, z * 0.5 + 2.0))
    # u 和 v 都用 wraparound,完全闭合
    for i in range(su):
        for j in range(sv):
            c   = i * sv + j
            nj  = i * sv + (j + 1) % sv
            ni  = ((i + 1) % su) * sv + j
            nij = ((i + 1) % su) * sv + (j + 1) % sv
            f.append((c, nj, ni))
            f.append((nj, nij, ni))
    return v, f

def gen_mobius_strip(R=1.0, w=0.4, steps=64, strips=10):
    """莫比乌斯带:u 和 v 两个方向都闭合(v 方向 strips 而非 strips+1,
    首尾共享同一条边,这是莫比乌斯带只有一条边的特性)"""
    v, f = [], []
    for i in range(steps):
        u = 2 * math.pi * i / steps
        cu2  = math.cos(u / 2)
        su_2 = math.sin(u / 2)
        cu   = math.cos(u)
        su   = math.sin(u)
        for j in range(strips):   # 注意:strips,不是 strips+1
            t = -w + 2 * w * j / strips
            r = R + t * cu2
            x = r * cu
            y = r * su
            z = t * su_2 + R + w + 0.1
            v.append((x, y, z))
    # u 方向 wraparound(steps-1 → steps)
    for i in range(steps):
        inext = (i + 1) % steps
        for j in range(strips):
            jnext = (j + 1) % strips
            c  = i      * strips + j
            cn = inext  * strips + j
            cj = i      * strips + jnext
            cjn = inext * strips + jnext
            f.append((c, cn, cjn))
            f.append((c, cjn, cj))
    return v, f

def gen_triple_torus():
    """三环面（亏格3）"""
    v1, f1 = gen_torus(0.55, 0.22, 18, 12)
    v2, f2 = gen_torus(0.55, 0.22, 18, 12)
    v3, f3 = gen_torus(0.55, 0.22, 18, 12)
    off = 1.4
    v2 = [(x+off, y, z) for x,y,z in v2]
    f2 = [(a+len(v1),b+len(v1),c+len(v1)) for a,b,c in f2]
    v3 = [(x+2*off, y, z) for x,y,z in v3]
    f3 = [(a+len(v1)+len(v2),b+len(v1)+len(v2),c+len(v1)+len(v2)) for a,b,c in f3]
    return v1+v2+v3, f1+f2+f3

# ===================== 动物 =====================
def gen_rabbit():
    v, f = [], []
    add_ellipsoid(v, f, 0, 0, 0.7, 0.5, 0.4, 0.6, 10, 12)         # 身体 (前后向 Y,头朝 +Y)
    add_ellipsoid(v, f, 0, 0.4, 1.4, 0.25, 0.25, 0.25, 8, 10)    # 头 (前上方,与身体略微重叠)
    for side in [-1, 1]:  # 耳朵 (头顶)
        add_ellipsoid(v, f, side*0.1, 0.4, 1.85, 0.04, 0.04, 0.2, 6, 6)
    add_ellipsoid(v, f, 0, -0.45, 0.55, 0.1, 0.1, 0.1, 5, 6)     # 尾巴 (尾部)
    for fx, fy in [(-0.2,0.2),(0.2,0.2),(-0.2,-0.2),(0.2,-0.2)]:  # 脚
        add_ellipsoid(v, f, fx, fy, 0.08, 0.1, 0.12, 0.08, 4, 6)
    return ensure_upright(v), f

def gen_cat():
    v, f = [], []
    add_ellipsoid(v, f, 0, 0, 0.55, 0.3, 0.45, 0.35, 10, 12)  # 身体
    add_ellipsoid(v, f, 0, 0.4, 1.0, 0.22, 0.2, 0.22, 8, 10)   # 头
    for s in [-1, 1]:  # 耳朵
        b = len(v)
        v.append((s*0.12, 0.4, 1.3)); v.append((s*0.06, 0.35, 1.15))
        v.append((s*0.2, 0.35, 1.15)); v.append((s*0.12, 0.5, 1.15))
        f.append((b,b+1,b+2)); f.append((b,b+2,b+3)); f.append((b,b+3,b+1))
    # 尾巴
    pts = [(0,-0.45,0.4),(0,-0.6,0.6),(0,-0.65,0.9),(0,-0.55,1.1)]
    add_tube(v, f, pts, 0.05, 6)
    for lx, ly in [(-0.12,0.2),(0.12,0.2),(-0.12,-0.2),(0.12,-0.2)]:
        add_cylinder(v, f, lx, ly, 0, 0.3, 0.05, 6, True)
    return ensure_upright(v), f

def gen_bird():
    v, f = [], []
    add_ellipsoid(v, f, 0, 0, 0.9, 0.2, 0.3, 0.25, 8, 10)   # 身体
    add_ellipsoid(v, f, 0, 0.25, 1.2, 0.13, 0.13, 0.13, 6, 8) # 头
    b = len(v)  # 喙
    v.append((0,0.42,1.2)); v.append((-0.03,0.3,1.17)); v.append((0.03,0.3,1.17))
    f.append((b,b+1,b+2))
    for s in [-1, 1]:  # 翅膀
        w = len(v)
        v.append((s*0.2,0,0.95)); v.append((s*0.6,0,0.9))
        v.append((s*0.2,-0.1,0.75)); v.append((s*0.5,-0.05,0.8))
        f.append((w,w+1,w+3)); f.append((w,w+3,w+2))
    t = len(v)  # 尾
    v.append((0,-0.3,0.8)); v.append((-0.08,-0.55,0.85)); v.append((0.08,-0.55,0.85))
    f.append((t,t+1,t+2))
    for s in [-1, 1]:  # 腿
        add_cylinder(v, f, s*0.06, 0, 0, 0.65, 0.02, 5, False)
    return ensure_upright(v), f

def gen_fish():
    """水平鱼:长轴沿 +X(头在右),躺在 z=0 平面上。"""
    v, f = [], []
    seg, rings = 14, 12
    start = len(v)
    for i in range(rings+1):
        t = i/rings
        x = 0.2 + t*1.6  # 头在 +x
        r_scale = math.sin(math.pi*t)*0.28 + 0.02
        for j in range(seg):
            theta = 2*math.pi*j/seg
            # 截面:y 侧向(宽),z 上下(扁)
            v.append((x, r_scale*math.sin(theta), r_scale*0.7*math.cos(theta)))
    for i in range(rings):
        for j in range(seg):
            c = start+i*seg+j; n = start+i*seg+(j+1)%seg
            cn = start+(i+1)*seg+j; nn = start+(i+1)*seg+(j+1)%seg
            f.append((c,n,cn)); f.append((n,nn,cn))
    # 尾鳍(在 xz 平面,垂直方向扇形)
    tf = len(v)
    v.append((0.2, 0, 0))        # 鱼尾端
    v.append((-0.15, 0, 0.3))    # 后上
    v.append((-0.15, 0, -0.05))  # 后下 (略低于鱼身)
    f.append((tf, tf+1, tf+2))
    # 背鳍(在 xy 平面,鱼背上方)
    df = len(v)
    v.append((1.1, 0, 0.22))
    v.append((1.3, 0, 0.45))
    v.append((0.9, 0, 0.22))
    f.append((df, df+1, df+2))
    return ensure_upright(v), f

def gen_deer():
    v, f = [], []
    add_ellipsoid(v, f, 0, 0, 1.0, 0.35, 0.6, 0.35, 10, 12)  # 身体
    add_ellipsoid(v, f, 0, 0.55, 1.5, 0.15, 0.15, 0.2, 6, 8)   # 头
    add_cylinder(v, f, 0, 0.35, 1.1, 1.5, 0.08, 8, False)       # 脖子
    for s in [-1, 1]:  # 鹿角
        pts = [(s*0.08,0.55,1.7),(s*0.15,0.55,2.0),(s*0.2,0.5,2.2)]
        add_tube(v, f, pts, 0.02, 5)
        pts2 = [(s*0.15,0.55,2.0),(s*0.25,0.6,2.1)]
        add_tube(v, f, pts2, 0.015, 4)
    for lx, ly in [(-0.15,0.35),(0.15,0.35),(-0.15,-0.35),(0.15,-0.35)]:
        add_cylinder(v, f, lx, ly, 0, 0.7, 0.05, 6, False)
    return ensure_upright(v), f

def gen_turtle():
    v, f = [], []
    add_ellipsoid(v, f, 0, 0, 0.35, 0.5, 0.4, 0.25, 10, 14)  # 壳
    add_ellipsoid(v, f, 0, 0.45, 0.25, 0.12, 0.12, 0.1, 5, 6) # 头
    for s in [-1, 1]:
        for fy in [0.2, -0.2]:
            add_ellipsoid(v, f, s*0.4, fy, 0.1, 0.12, 0.06, 0.06, 4, 5)
    add_ellipsoid(v, f, 0, -0.45, 0.2, 0.05, 0.08, 0.04, 4, 5)  # 尾巴
    return ensure_upright(v), f

# ===================== 植物 =====================
def gen_tree():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 1.0, 0.12, 8, True)
    for radius, height, base_z in [(0.8,1.0,0.9),(0.6,0.9,1.5),(0.4,0.8,2.0)]:
        add_cone(v, f, 0, 0, base_z, base_z+height, radius, 10)
    return v, f

def gen_pine_tree():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 0.8, 0.1, 8, True)
    for r, h, bz in [(0.7,0.7,0.7),(0.55,0.7,1.2),(0.4,0.7,1.7),(0.25,0.6,2.1)]:
        add_cone(v, f, 0, 0, bz, bz+h, r, 8)
    return v, f

def gen_flower():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 1.0, 0.03, 6, False)
    for p in range(5):
        ang = 2*math.pi*p/5
        b = len(v)
        cx, cy = math.cos(ang)*0.15, math.sin(ang)*0.15
        v.append((0,0,1.0)); v.append((cx-0.06*math.sin(ang),cy+0.06*math.cos(ang),1.02))
        v.append((cx*2.5, cy*2.5, 1.05)); v.append((cx+0.06*math.sin(ang),cy-0.06*math.cos(ang),1.02))
        f.append((b,b+1,b+2)); f.append((b,b+2,b+3))
    add_ellipsoid(v, f, 0, 0, 1.03, 0.06, 0.06, 0.05, 4, 6)
    return v, f

def gen_cactus():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 1.2, 0.2, 10, True)
    add_ellipsoid(v, f, 0, 0, 1.2, 0.2, 0.2, 0.1, 5, 10)
    for side, az in [(-1, 0.6), (1, 0.8)]:
        add_cylinder(v, f, side*0.3, 0, az, az, 0.1, 8, False)  # 水平连接
        add_cylinder(v, f, side*0.4, 0, az, az+0.5, 0.1, 8, True)
    return v, f

def gen_mushroom():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 0.6, 0.1, 10, True)  # 柄
    # 伞盖（半球）
    start = len(v); seg, rings = 14, 8; R = 0.4
    for r in range(rings+1):
        phi = math.pi*0.5*r/rings
        for s in range(seg):
            theta = 2*math.pi*s/seg
            v.append((R*math.cos(phi)*math.cos(theta), R*math.cos(phi)*math.sin(theta), 0.6+R*math.sin(phi)))
    for r in range(rings):
        for s in range(seg):
            c = start+r*seg+s; n = start+r*seg+(s+1)%seg
            cn = start+(r+1)*seg+s; nn = start+(r+1)*seg+(s+1)%seg
            f.append((c,n,cn)); f.append((n,nn,cn))
    return v, f

# ===================== 工具 =====================
def gen_hammer():
    v, f = [], []
    add_box(v, f, 0, 0, 0.6, 0.04, 0.04, 0.6)   # 柄
    add_box(v, f, 0, 0, 1.28, 0.25, 0.08, 0.1)    # 锤头
    return v, f

def gen_wrench():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 1.0, 0.04, 8, True)
    for s in [-1, 1]:
        add_box(v, f, s*0.07, 0, 1.12, 0.03, 0.1, 0.12)
    return v, f

def gen_screwdriver():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 0.5, 0.08, 10, True)   # 手柄
    add_cylinder(v, f, 0, 0, 0.5, 1.2, 0.025, 8, False) # 杆
    add_box(v, f, 0, 0, 1.24, 0.04, 0.008, 0.04)        # 扁头
    return v, f

def gen_pliers():
    v, f = [], []
    for s in [-1, 1]:
        add_box(v, f, s*0.04, 0, 0.35, 0.025, 0.02, 0.35)  # 手柄
        add_box(v, f, s*0.03, 0, 0.85, 0.02, 0.03, 0.15)    # 钳口
    add_cylinder(v, f, 0, 0, 0.68, 0.72, 0.04, 8, True)      # 铰接点
    return v, f

def gen_saw():
    v, f = [], []
    add_box(v, f, 0, 0, 0.38, 0.5, 0.005, 0.08)  # 锯片
    add_box(v, f, -0.55, 0, 0.35, 0.04, 0.03, 0.12)  # 手柄
    return v, f

def gen_axe():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 1.2, 0.035, 8, True)  # 柄
    # 斧头（楔形）
    b = len(v)
    v += [(0,-0.01,1.0),(0,0.01,1.0),(0,-0.01,1.3),(0,0.01,1.3),
          (0.25,-0.005,1.05),(0.25,0.005,1.05),(0.25,-0.005,1.25),(0.25,0.005,1.25)]
    quads = [(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    for q in quads:
        f.append((b+q[0],b+q[1],b+q[2])); f.append((b+q[0],b+q[2],b+q[3]))
    return v, f

# ===================== 建筑 =====================
def gen_house():
    v, f = [], []
    add_box(v, f, 0, 0, 0.5, 1.0, 0.8, 0.5)  # 墙
    # 屋顶
    b = len(v)
    w, d, h, rh = 1.15, 0.95, 1.0, 0.6
    v += [(-w,-d,h),(w,-d,h),(w,d,h),(-w,d,h),(0,-d,h+rh),(0,d,h+rh)]
    f.append((b,b+1,b+4)); f.append((b+1,b+5,b+4)); f.append((b+1,b+2,b+5))
    f.append((b+2,b+3,b+5)); f.append((b+3,b,b+4)); f.append((b+3,b+4,b+5))
    return v, f

def gen_tower():
    v, f = [], []
    add_cylinder(v, f, 0, 0, 0, 2.5, 0.5, 12, True)
    add_cone(v, f, 0, 0, 2.5, 3.3, 0.65, 12)
    return v, f

def gen_castle():
    v, f = [], []
    add_box(v, f, 0, 0, 0.6, 1.5, 1.5, 0.6)
    for cx, cy in [(-1.5,-1.5),(1.5,-1.5),(1.5,1.5),(-1.5,1.5)]:
        add_cylinder(v, f, cx, cy, 0, 1.8, 0.3, 8, True)
        add_cone(v, f, cx, cy, 1.8, 2.3, 0.38, 8)
    # 城垛
    for i in range(6):
        t = -1.2 + 2.4*i/5
        for side in [-1.5, 1.5]:
            add_box(v, f, t, side, 1.32, 0.12, 0.12, 0.12)
            add_box(v, f, side, t, 1.32, 0.12, 0.12, 0.12)
    return v, f

def gen_bridge():
    v, f = [], []
    add_box(v, f, 0, 0, 0.85, 2.0, 0.4, 0.05)  # 桥面
    seg = 14; arch_r = 0.7
    for side in [-0.4, 0.4]:
        base = len(v)
        for i in range(seg+1):
            ang = math.pi*i/seg
            x = -arch_r*math.cos(ang); z = arch_r*math.sin(ang)
            v.append((x, side, z)); v.append((x, side+0.05*(1 if side>0 else -1), z))
        for i in range(seg):
            bb = base+i*2
            f.append((bb,bb+1,bb+2)); f.append((bb+1,bb+3,bb+2))
    for px in [-1.6, 1.6]:
        add_box(v, f, px, 0, 0.42, 0.15, 0.4, 0.42)
    return v, f

def gen_pyramid_building():
    """金字塔"""
    v, f = [], []
    add_cone(v, f, 0, 0, 0, 2.0, 1.5, 4)
    return v, f

# ===================== 人物 =====================
def _human_base():
    """通用人形骨架"""
    v, f = [], []
    add_ellipsoid(v, f, 0, 0, 1.55, 0.12, 0.12, 0.14, 8, 10)  # 头
    add_cylinder(v, f, 0, 0, 1.35, 1.42, 0.05, 6, False)        # 脖子
    add_box(v, f, 0, 0, 1.05, 0.2, 0.12, 0.3)                   # 躯干
    return v, f

def gen_human_standing():
    v, f = _human_base()
    for s in [-1, 1]:
        add_cylinder(v, f, s*0.28, 0, 0.8, 1.3, 0.05, 6, False)  # 上臂
        add_cylinder(v, f, s*0.28, 0, 0.45, 0.8, 0.045, 6, False) # 前臂
        add_cylinder(v, f, s*0.1, 0, 0, 0.55, 0.07, 6, False)     # 大腿
        add_cylinder(v, f, s*0.1, 0, 0, 0.05, 0.06, 6, False)     # 小腿部分
        add_box(v, f, s*0.1, 0.04, 0.03, 0.06, 0.1, 0.03)         # 脚
    add_cylinder(v, f, -0.1, 0, 0.05, 0.55, 0.07, 6, False)
    add_cylinder(v, f, 0.1, 0, 0.05, 0.55, 0.07, 6, False)
    return ensure_upright(v), f

def gen_human_walking():
    v, f = _human_base()
    # 手臂摆动
    add_tube(v, f, [(0.28,0,1.3),(0.32,0.1,1.0),(0.3,0.15,0.75)], 0.045, 6)
    add_tube(v, f, [(-0.28,0,1.3),(-0.32,-0.1,1.0),(-0.3,-0.15,0.75)], 0.045, 6)
    # 腿迈步
    add_tube(v, f, [(0.1,0,0.75),(0.15,0.15,0.4),(0.12,0.2,0.05)], 0.06, 6)
    add_tube(v, f, [(-0.1,0,0.75),(-0.15,-0.12,0.4),(-0.12,-0.15,0.05)], 0.06, 6)
    add_box(v, f, 0.12, 0.24, 0.03, 0.05, 0.09, 0.03)
    add_box(v, f, -0.12, -0.19, 0.03, 0.05, 0.09, 0.03)
    return ensure_upright(v), f

def gen_human_running():
    v, f = _human_base()
    # 手臂大幅摆动
    add_tube(v, f, [(0.25,0,1.3),(0.35,0.2,1.1),(0.3,0.3,0.9)], 0.04, 6)
    add_tube(v, f, [(-0.25,0,1.3),(-0.35,-0.15,1.05),(-0.35,-0.25,0.85)], 0.04, 6)
    # 腿大幅迈步
    add_tube(v, f, [(0.1,0,0.75),(0.2,0.25,0.45),(0.15,0.35,0.1)], 0.06, 6)
    add_tube(v, f, [(-0.1,0,0.75),(-0.2,-0.2,0.5),(-0.25,-0.3,0.2)], 0.06, 6)
    add_box(v, f, 0.15, 0.39, 0.08, 0.05, 0.09, 0.03)
    add_box(v, f, -0.25, -0.34, 0.18, 0.05, 0.09, 0.03)
    return ensure_upright(v), f

def gen_human_sitting():
    v, f = [], []
    add_ellipsoid(v, f, 0, 0, 1.15, 0.12, 0.12, 0.14, 8, 10)  # 头
    add_box(v, f, 0, 0, 0.85, 0.2, 0.12, 0.2)                   # 躯干（较短）
    for s in [-1, 1]:
        add_tube(v, f, [(s*0.25,0,1.0),(s*0.28,0,0.75),(s*0.25,0.05,0.55)], 0.04, 6)  # 手臂
        # 大腿水平
        add_tube(v, f, [(s*0.1,0,0.65),(s*0.1,0.2,0.63),(s*0.1,0.4,0.6)], 0.065, 6)
        # 小腿垂直
        add_cylinder(v, f, s*0.1, 0.4, 0, 0.6, 0.055, 6, False)
        add_box(v, f, s*0.1, 0.44, 0.03, 0.05, 0.09, 0.03)
    return ensure_upright(v), f

def gen_human_waving():
    v, f = _human_base()
    # 右手举起
    add_tube(v, f, [(0.25,0,1.3),(0.35,0,1.5),(0.35,0,1.75)], 0.045, 6)
    add_ellipsoid(v, f, 0.35, 0, 1.8, 0.05, 0.05, 0.05, 4, 5)  # 手掌
    # 左手自然下垂
    add_tube(v, f, [(-0.28,0,1.3),(-0.3,0,1.0),(-0.28,0,0.7)], 0.045, 6)
    for s in [-1, 1]:
        add_cylinder(v, f, s*0.1, 0, 0, 0.55, 0.07, 6, False)
        add_cylinder(v, f, s*0.1, 0, 0.55, 0.75, 0.06, 6, False)
        add_box(v, f, s*0.1, 0.04, 0.03, 0.06, 0.1, 0.03)
    return ensure_upright(v), f

# ===================== 其他 =====================
def gen_cup():
    v, f = [], []
    seg = 14
    # 外壁（圆台）
    for i in range(seg):
        t = 2*math.pi*i/seg
        v.append((0.3*math.cos(t), 0.3*math.sin(t), 0))
    for i in range(seg):
        t = 2*math.pi*i/seg
        v.append((0.4*math.cos(t), 0.4*math.sin(t), 0.7))
    for i in range(seg):
        ni = (i+1)%seg
        f.append((i,ni,seg+i)); f.append((ni,seg+ni,seg+i))
    # 底
    bc = len(v); v.append((0,0,0))
    for i in range(seg):
        ni = (i+1)%seg
        f.append((bc,ni,i))
    # 把手
    pts = [(0.4,0,0.55),(0.55,0,0.5),(0.58,0,0.35),(0.5,0,0.2),(0.38,0,0.15)]
    add_tube(v, f, pts, 0.03, 6)
    return v, f

def gen_chair():
    v, f = [], []
    add_box(v, f, 0, 0, 0.45, 0.25, 0.25, 0.03)  # 坐面
    add_box(v, f, 0, -0.23, 0.75, 0.25, 0.03, 0.3) # 靠背
    for lx, ly in [(-0.2,-0.2),(0.2,-0.2),(-0.2,0.2),(0.2,0.2)]:
        add_cylinder(v, f, lx, ly, 0, 0.42, 0.025, 6, False)
    return v, f

def gen_table():
    v, f = [], []
    add_box(v, f, 0, 0, 0.75, 0.6, 0.4, 0.04)
    for lx, ly in [(-0.5,-0.3),(0.5,-0.3),(-0.5,0.3),(0.5,0.3)]:
        add_cylinder(v, f, lx, ly, 0, 0.71, 0.035, 6, False)
    return v, f

def gen_car():
    v, f = [], []
    add_box(v, f, 0, 0, 0.35, 0.9, 0.4, 0.2)    # 车身
    add_box(v, f, -0.1, 0, 0.65, 0.45, 0.35, 0.15) # 车顶
    for wx, wy in [(-0.55,-0.4),(-0.55,0.4),(0.55,-0.4),(0.55,0.4)]:
        add_cylinder(v, f, wx, wy, 0.15, 0.15, 0.15, 10, True)  # 轮子（简化）
        # 让轮子竖起来 - 用椭球代替
    return ensure_upright(v), f

def gen_airplane():
    v, f = [], []
    # 机身
    add_ellipsoid(v, f, 0, 0, 1.0, 0.15, 0.15, 0.8, 8, 12)
    # 机翼
    for s in [-1, 1]:
        w = len(v)
        v.append((0,0,1.0)); v.append((s*1.2,0,0.95)); v.append((s*1.0,0,0.85)); v.append((0,0,0.85))
        f.append((w,w+1,w+2)); f.append((w,w+2,w+3))
    # 尾翼
    t = len(v)
    v.append((0,0,0.3)); v.append((0,0.4,0.15)); v.append((0,0,0.15))
    f.append((t,t+1,t+2))
    t2 = len(v)
    v.append((0,0,0.3)); v.append((-0.3,0,0.2)); v.append((0.3,0,0.2))
    f.append((t2,t2+1,t2+2))
    return ensure_upright(v), f

def gen_boat():
    v, f = [], []
    # 船体（简化为梯形截面拉伸）
    seg = 10
    for i in range(seg+1):
        t = i/seg
        z = 0.1 + 0.2*math.sin(math.pi*t)
        hw = 0.3*(1-0.3*abs(t-0.5)*2)
        v.append((-1.0+2*t, -hw, z)); v.append((-1.0+2*t, hw, z))
        v.append((-1.0+2*t, -hw*0.6, 0)); v.append((-1.0+2*t, hw*0.6, 0))
    for i in range(seg):
        b = i*4
        for row in [(0,1),(2,3)]:
            f.append((b+row[0],b+4+row[0],b+row[1]))
            f.append((b+row[1],b+4+row[0],b+4+row[1]))
        f.append((b+0,b+2,b+6)); f.append((b+0,b+6,b+4))
        f.append((b+1,b+5,b+7)); f.append((b+1,b+7,b+3))
    # 桅杆+帆
    add_cylinder(v, f, 0, 0, 0.3, 1.5, 0.02, 5, False)
    s = len(v)
    v.append((0,0,1.4)); v.append((0,0,0.5)); v.append((0.6,0,0.6))
    f.append((s,s+1,s+2))
    return ensure_upright(v), f

def gen_star():
    """五角星（立体）"""
    v, f = [], []
    outer, inner, h = 1.0, 0.4, 0.15
    pts_top, pts_bot = [], []
    for i in range(10):
        ang = math.pi/2 + 2*math.pi*i/10
        r = outer if i%2==0 else inner
        pts_top.append((r*math.cos(ang), r*math.sin(ang), h))
        pts_bot.append((r*math.cos(ang), r*math.sin(ang), 0))
    v.extend(pts_bot); v.extend(pts_top)
    # 侧面
    for i in range(10):
        ni = (i+1)%10
        f.append((i, ni, 10+i)); f.append((ni, 10+ni, 10+i))
    # 顶面扇形
    ct = len(v); v.append((0,0,h))
    cb = len(v); v.append((0,0,0))
    for i in range(10):
        ni = (i+1)%10
        f.append((ct, 10+i, 10+ni))
        f.append((cb, ni, i))
    return v, f

def gen_heart():
    """心形（3D）"""
    v, f = [], []
    rings, sectors = 20, 24
    for i in range(rings+1):
        t = math.pi * i / rings
        for j in range(sectors):
            theta = 2*math.pi*j/sectors
            # 心形参数方程
            r = 0.5*(1 - math.cos(t))
            x = r * math.cos(theta) * 0.5
            y = r * math.sin(theta) * 0.5
            z = 0.5*(math.cos(t) - 1) * 0.3 + (16*math.sin(t/2)**4)*0.05
            # 使用经典心形曲线旋转
            hx = 16*math.sin(t)**3
            hy = 13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t)
            scale = 0.04
            rr = abs(hx)*scale*0.3
            z_val = hy*scale + 1.0
            x_val = rr*math.cos(theta)
            y_val = rr*math.sin(theta)
            v.append((x_val, y_val, z_val))
    for i in range(rings):
        for j in range(sectors):
            c = i*sectors+j; n = i*sectors+(j+1)%sectors
            cn = (i+1)*sectors+j; nn = (i+1)*sectors+(j+1)%sectors
            f.append((c,n,cn)); f.append((n,nn,cn))
    return ensure_upright(v), f

# ===================== 主函数 =====================
def main():
    models = {
        "01_正多面体": {
            "正四面体": gen_tetrahedron,
            "正六面体": gen_cube,
            "正八面体": gen_octahedron,
            "正十二面体": gen_dodecahedron,
            "正二十面体": gen_icosahedron,
        },
        "02_凸多面体": {
            "截角四面体": gen_truncated_tetrahedron,
            "截角立方体": gen_truncated_cube,
            "立方八面体": gen_cuboctahedron,
            "二十-十二面体": gen_icosidodecahedron,
            "菱形立方八面体": gen_rhombicuboctahedron,
        },
        "03_球体与锥体": {
            "球体": gen_sphere,
            "圆锥": gen_cone_shape,
            "三棱锥": gen_pyramid_3,
            "四棱锥": gen_pyramid_4,
            "六棱锥": gen_pyramid_6,
            "圆台": gen_frustum,
        },
        "04_多亏格曲面": {
            "环面_亏格1": gen_torus,
            "双环面_亏格2": gen_double_torus,
            "三环面_亏格3": gen_triple_torus,
            "克莱因瓶": gen_klein_bottle,
            "莫比乌斯带": gen_mobius_strip,
        },
        "05_动物": {
            "兔子": gen_rabbit,
            "猫": gen_cat,
            "鸟": gen_bird,
            "鱼": gen_fish,
            "鹿": gen_deer,
            "乌龟": gen_turtle,
        },
        "06_植物": {
            "阔叶树": gen_tree,
            "松树": gen_pine_tree,
            "花": gen_flower,
            "仙人掌": gen_cactus,
            "蘑菇": gen_mushroom,
        },
        "07_工具": {
            "锤子": gen_hammer,
            "扳手": gen_wrench,
            "螺丝刀": gen_screwdriver,
            "钳子": gen_pliers,
            "手锯": gen_saw,
            "斧头": gen_axe,
        },
        "08_建筑": {
            "房屋": gen_house,
            "塔楼": gen_tower,
            "城堡": gen_castle,
            "拱桥": gen_bridge,
            "金字塔": gen_pyramid_building,
        },
        "09_人物动作": {
            "站立": gen_human_standing,
            "行走": gen_human_walking,
            "奔跑": gen_human_running,
            "坐姿": gen_human_sitting,
            "挥手": gen_human_waving,
        },
        "10_其他": {
            "杯子": gen_cup,
            "椅子": gen_chair,
            "桌子": gen_table,
            "汽车": gen_car,
            "飞机": gen_airplane,
            "小船": gen_boat,
            "五角星": gen_star,
            "心形": gen_heart,
        },
    }

    total = 0
    for folder, items in models.items():
        for name, gen_func in items.items():
            try:
                verts, faces = gen_func()
                # 合并空间上接近的顶点(消除拼接处的裂缝)
                verts, faces = weld_vertices(verts, faces, tol=1e-3)
                path = BASE_DIR / folder / f"{name}.ply"
                write_ply(path, verts, faces)
                total += 1
                print(f"  [OK] {folder}/{name}.ply  ({len(verts)} verts, {len(faces)} faces)")
            except Exception as e:
                print(f"  [FAIL] {folder}/{name}.ply -> {e}")
    print(f"\n完成! 共生成 {total} 个 PLY 模型文件 -> {BASE_DIR}")

if __name__ == "__main__":
    main()
