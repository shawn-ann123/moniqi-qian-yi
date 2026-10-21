#!/usr/bin/env python3
# 离线解算：quat 存储顺序 × 相机轴向映射 全枚举，与橙光斑质心匹配
import math, itertools, sys
from PIL import Image

DUMP = "/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/dump/dump.txt"
FMT  = "/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/dump/f_%+04d.ppm"
FX = FY = 1303.67532368147
CX, CY = 720.0, 540.0

def quat_to_R(q, order):
    if order == 'wxyz':
        w, x, y, z = q
    else:
        x, y, z, w = q
    return [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]

def mat_vec(M, v):
    return [sum(M[i][j]*v[j] for j in range(3)) for i in range(3)]

def transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]

def signed_perms():
    """48 个带符号置换矩阵 (含镜像)"""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1,-1], repeat=3):
            M = [[0]*3 for _ in range(3)]
            for i in range(3):
                M[i][perm[i]] = signs[i]
            out.append((perm, signs, M))
    return out

def detect_orange(path):
    img = Image.open(path)
    W, H = img.size
    px = img.load()
    sx = sy = 0.0; n = 0
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            r, g, b = px[x, y]
            if r > 180 and r-b > 80 and r-g > 40:
                sx += x; sy += y; n += 1
    if n < 10:
        return None
    return (sx/n, sy/n, n)

# 解析 dump
frames = []
with open(DUMP) as f:
    for line in f:
        if line.startswith('#'): continue
        tok = line.split()
        vals = [float(t) for t in tok]
        yaw = int(vals[0])
        seq = int(vals[2])
        cam = vals[3:6]
        q = vals[6:10]
        ch_rpy = vals[13:16]
        g_yaw, g_pitch = vals[16], vals[17]
        idx = 18
        rc = int(vals[idx]); idx += 1
        runes = []
        for _ in range(rc):
            runes.append(vals[idx:idx+7]); idx += 7
        tc = int(vals[idx]); idx += 1
        targets = []
        for _ in range(tc):
            targets.append(vals[idx:idx+7]); idx += 7
        frames.append(dict(yaw=yaw, seq=seq, cam=cam, q=q, ch_rpy=ch_rpy,
                           g_yaw=g_yaw, g_pitch=g_pitch, runes=runes, targets=targets))

print(f"frames: {len(frames)}")

# 检测光斑
MANUAL_HUB = {30: (605, 700), 45: (920, 682), 60: (927, 687), 75: (1327, 673)}
for fr in frames:
    if fr['yaw'] in MANUAL_HUB:
        u, v = MANUAL_HUB[fr['yaw']]
        fr['blob'] = (float(u), float(v), 999)
    else:
        fr['blob'] = detect_orange(FMT % fr['yaw'])
found = [fr for fr in frames if fr['blob']]
print(f"frames with rune hub: {len(found)} of {len(frames)}")
for fr in found:
    print(f"  yaw={fr['yaw']:+04d} hub=({fr['blob'][0]:.0f},{fr['blob'][1]:.0f}) g_yaw={fr['g_yaw']:.3f}")

# 用 target[0] (自车) 验证 odom==world:
chk = frames[0]
print("\nchassis pos:", chk['cam'], " target0:", chk['targets'][0][:3])

# 枚举假设
best = []
for order in ('wxyz', 'xyzw'):
    for pi, (perm, signs, M) in enumerate(signed_perms()):
        errs = []
        for fr in found:
            R = quat_to_R(fr['q'], order)
            Rt = transpose(R)
            # rune[0]
            P = fr['runes'][0][:3]
            d = [P[i]-fr['cam'][i] for i in range(3)]
            pm = mat_vec(Rt, d)      # 数学系
            po = mat_vec(M, pm)      # optical 假设
            if po[2] <= 0.01:
                errs.append(10000)
                continue
            u = FX*po[0]/po[2] + CX
            v = FY*po[1]/po[2] + CY
            u0, v0, _ = fr['blob']
            errs.append(math.hypot(u-u0, v-v0))
        best.append((sum(errs)/len(errs), order, perm, signs))

best.sort(key=lambda t: t[0])
print("\n== top10 假设 (平均像素误差) ==")
for e, order, perm, signs in best[:10]:
    print(f"  err={e:9.1f}px order={order} perm={perm} signs={signs}")

# 细节: 最优假设逐帧投影
e, order, perm, signs = best[0]
print(f"\n== 最优假设逐帧 ==  (runes[0] 橙队)")
for fr in found:
    R = quat_to_R(fr['q'], order)
    Rt = transpose(R)
    M = [[0]*3 for _ in range(3)]
    for i in range(3): M[i][perm[i]] = signs[i]
    row = []
    for ri in range(2):
        P = fr['runes'][ri][:3]
        d = [P[i]-fr['cam'][i] for i in range(3)]
        po = mat_vec(M, mat_vec(Rt, d))
        if po[2] > 0.01:
            u = FX*po[0]/po[2] + CX; v = FY*po[1]/po[2] + CY
            row.append(f"r{ri}=({u:.0f},{v:.0f})")
        else:
            row.append(f"r{ri}=behind")
    b = fr['blob']
    print(f"  yaw={fr['yaw']:+04d} blob=({b[0]:.0f},{b[1]:.0f}) " + " ".join(row))

# 对 top 假设做偏移网格搜索 (r_center 真值可能不是轮毂中心)
print("\n== top5 假设 + 偏移网格搜索 ==")
import itertools
OFF = [x*0.2 for x in range(-5, 6)]
for e0, order, perm, signs in best[:5]:
    M = [[0]*3 for _ in range(3)]
    for i in range(3): M[i][perm[i]] = signs[i]
    best_off = (1e18, None)
    for dx in OFF:
        for dy in OFF:
            for dz in OFF:
                errs = []
                for fr in found:
                    R = quat_to_R(fr['q'], order)
                    Rt = transpose(R)
                    P = [fr['runes'][0][i] + (dx, dy, dz)[i] for i in range(3)]
                    d = [P[i]-fr['cam'][i] for i in range(3)]
                    po = mat_vec(M, mat_vec(Rt, d))
                    if po[2] <= 0.01:
                        errs.append(10000); continue
                    u = FX*po[0]/po[2] + CX; v = FY*po[1]/po[2] + CY
                    u0, v0, _ = fr['blob']
                    errs.append(math.hypot(u-u0, v-v0))
                m = sum(errs)/len(errs)
                if m < best_off[0]:
                    best_off = (m, (round(dx,2), round(dy,2), round(dz,2)))
    print(f"  base_err={e0:8.1f}px order={order} perm={perm} signs={signs} -> best_err={best_off[0]:7.1f}px off={best_off[1]}")
