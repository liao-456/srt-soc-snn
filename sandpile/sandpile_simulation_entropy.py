import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sizes = [20, 30, 50]        # 网格尺寸
Zc = 4
total_steps = 10000
window_size = 500           # 熵窗口
seed = 42                   

def run_simulation_async(L, Zc=4, total_steps=10000, seed=42, verbose=True):
    """异步序贯更新的BTW沙堆模型。
    每次随机选取一个不稳定格崩塌并更新邻居，直至全网格稳定。
    依据Dhar可交换性，雪崩总规模与崩塌顺序无关，随机选取顺序不影响统计结果。"""
    rng = np.random.default_rng(seed)
    grid = np.zeros((L, L), dtype=int)
    av_sizes = []
    records = []

    for step in range(total_steps):
        i, j = rng.integers(0, L, size=2)
        grid[i, j] += 1                      # 驱动：随机加一粒沙

        size = 0
        if grid[i, j] >= Zc:
            stack = [(i, j)]                 # 不稳定格栈
            while stack:
                k = rng.integers(0, len(stack))   # 随机挑一个
                i0, j0 = stack[k]
                stack[k] = stack[-1]         # 与栈尾交换后弹出
                stack.pop()

                grid[i0, j0] -= Zc           # 崩塌
                size += 1
                if grid[i0, j0] >= Zc:       # 仍不稳定，重新入栈
                    stack.append((i0, j0))

                for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ni, nj = i0 + di, j0 + dj
                    if 0 <= ni < L and 0 <= nj < L:
                        grid[ni, nj] += 1
                        if grid[ni, nj] == Zc:   # 首次达到阈值才入栈，避免重复
                            stack.append((ni, nj))

        if size > 0:
            av_sizes.append(size)
            records.append((step, size))

        if verbose and (step + 1) % 2000 == 0:
            print(f"L={L}: {step+1}/{total_steps} 完成, 雪崩 {len(av_sizes)} 次")

    return av_sizes, records

results = {}
all_records = {}

for L in sizes:
    t0 = time.time()
    av_sizes, rec = run_simulation_async(L, Zc, total_steps, seed)
    results[L] = {
        'sizes': av_sizes,
        'count': len(av_sizes),
        'mean': float(np.mean(av_sizes)) if av_sizes else 0.0,
        'max': max(av_sizes) if av_sizes else 0
    }
    all_records[L] = rec
    print(f"L={L}: 雪崩{len(av_sizes)}次, 平均{results[L]['mean']:.2f}, "
          f"最大{results[L]['max']}, 耗时{time.time()-t0:.1f}s")

    def calculate_entropy(sizes_list):
    if len(sizes_list) == 0:
        return 0.0
    values, counts = np.unique(sizes_list, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))      # 自然对数

def entropy_evolution(records, total_steps, window_size=500):
    times, ent = [], []
    for start in range(0, total_steps, window_size):
        end = start + window_size
        in_win = [s for (st, s) in records if start <= st < end]
        times.append(start)
        ent.append(calculate_entropy(in_win))
    return times, ent

plt.figure(figsize=(12, 6))
colors = {20: 'b', 30: 'g', 50: 'r'}
for L in sizes:
    t, e = entropy_evolution(all_records[L], total_steps, window_size)
    plt.plot(t, e, color=colors[L], marker='o', markersize=3, linewidth=2, label=f'{L}×{L}')
plt.xlabel('模拟时间步')
plt.ylabel('香农熵 H (nat)')
plt.title('不同网格尺寸下熵变曲线对比（异步更新）')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('entropy_async.png', dpi=300, bbox_inches='tight'); plt.show()

def fit_powerlaw_pdf(sizes, L):
    max_s = np.max(sizes)
    bins = np.logspace(np.log10(1), np.log10(max_s), 30)   # 30个对数箱
    counts, edges = np.histogram(sizes, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    pdf = counts / (len(sizes) * widths)
    nonzero = counts > 0
    x, y = centers[nonzero], pdf[nonzero]
    m = (x >= 5) & (x <= max_s * 0.5)                     # 拟合区间与原代码一致
    if m.sum() < 5:
        return None, None
    slope, intercept, r, p, se = linregress(np.log(x[m]), np.log(y[m]))
    return -slope, r**2

print(f"{'尺寸':<10}{'雪崩次数':<10}{'平均规模':<10}{'最大规模':<10}{'τ(PDF)':<10}{'R²':<10}")
for L in sizes:
    tau, r2 = fit_powerlaw_pdf(results[L]['sizes'], L)
    print(f"{L}×{L:<7}{results[L]['count']:<10}{results[L]['mean']:<10.2f}"
          f"{results[L]['max']:<10}{tau:<10.3f}{r2:<10.3f}")

    import numpy as np
from scipy import stats

L = 20
Zc = 4
TOTAL = 50000
SEED = 42          # 与沙堆主实验一致

def evolve_async(L, Zc, total_steps, seed, snap_steps=(1000, 45000)):
    rng = np.random.default_rng(seed)
    grid = np.zeros((L, L), dtype=int)
    snaps = {}
    for step in range(1, total_steps + 1):
        i, j = rng.integers(0, L, size=2)
        grid[i, j] += 1
        stack = [(i, j)] if grid[i, j] >= Zc else []
        while stack:
            k = rng.integers(0, len(stack))
            i0, j0 = stack[k]; stack[k] = stack[-1]; stack.pop()
            if grid[i0, j0] < Zc:
                continue
            grid[i0, j0] -= Zc
            if grid[i0, j0] >= Zc:
                stack.append((i0, j0))
            for di, dj in ((0,1),(0,-1),(1,0),(-1,0)):
                ni, nj = i0+di, j0+dj
                if 0 <= ni < L and 0 <= nj < L:
                    grid[ni, nj] += 1
                    if grid[ni, nj] == Zc:
                        stack.append((ni, nj))
        if step in snap_steps:
            snaps[step] = grid.copy()
    return snaps

snaps = evolve_async(L, Zc, TOTAL, SEED)
for k, g in snaps.items():
    print(f"第{k}步快照: 平均高度 {g.mean():.3f}")

def stimulate(snapshot, strength=2, trials=500, seed=0):
    rng = np.random.default_rng(seed)
    c = L // 2
    sizes, depths = [], []
    for _ in range(trials):
        grid = snapshot.copy()
        grid[c, c] += strength
        size, depth = 0, -1
        stack = [(c, c, 0)] if grid[c, c] >= Zc else []
        while stack:
            k = rng.integers(0, len(stack))
            i0, j0, d = stack[k]; stack[k] = stack[-1]; stack.pop()
            if grid[i0, j0] < Zc:
                continue
            grid[i0, j0] -= Zc
            size += 1
            depth = max(depth, d)
            if grid[i0, j0] >= Zc:
                stack.append((i0, j0, d))
            for di, dj in ((0,1),(0,-1),(1,0),(-1,0)):
                ni, nj = i0+di, j0+dj
                if 0 <= ni < L and 0 <= nj < L:
                    grid[ni, nj] += 1
                    if grid[ni, nj] == Zc:
                        stack.append((ni, nj, d+1))
        sizes.append(size)
        depths.append(depth + 1 if size > 0 else 0)
    return np.array(sizes), np.array(depths)

def report(name, s, d):
    nz = s > 0
    print(f"{name}: 非零比例 {nz.mean()*100:.1f}%, 平均规模 {s.mean():.2f}, "
          f"最大规模 {s.max()}, P90={np.percentile(s,90):.0f}, P99={np.percentile(s,99):.0f}, "
          f"平均深度 {d[nz].mean():.2f}")

s_sub, d_sub = stimulate(snaps[1000])
s_crit, d_crit = stimulate(snaps[45000])
report("第1000步(暂态)", s_sub, d_sub)
report("第45000步(临界)", s_crit, d_crit)

t, p = stats.ttest_ind(s_sub, s_crit, equal_var=False)
print(f"平均规模 Welch t检验: t={t:.2f}, p={p:.3f}")


import numpy as np
from scipy import stats

L, Zc = 20, 4

def relax(rng, grid):
    """对给定网格执行异步弛豫至稳定"""
    # 找出所有不稳定格（简化入口，直接全扫描）
    while True:
        unst = np.argwhere(grid >= Zc)
        if len(unst) == 0:
            break
        k = rng.integers(0, len(unst))
        i0, j0 = unst[k]
        grid[i0, j0] -= Zc
        for di, dj in ((0,1),(0,-1),(1,0),(-1,0)):
            ni, nj = i0+di, j0+dj
            if 0 <= ni < L and 0 <= nj < L:
                grid[ni, nj] += 1

def evolve(L, Zc, total_steps, seed):
    rng = np.random.default_rng(seed)
    grid = np.zeros((L, L), dtype=int)
    for _ in range(total_steps):
        i, j = rng.integers(0, L, size=2)
        grid[i, j] += 1
        if grid[i, j] >= Zc:
            relax(rng, grid)
    return grid

# 暂态系综：200次独立短演化，各取第1000步快照
transient_snaps = []
for s in range(200):
    transient_snaps.append(evolve(L, Zc, 1000, seed=1000 + s))

# 临界系综：1次长演化，每100步存快照
rng = np.random.default_rng(42)
grid = np.zeros((L, L), dtype=int)
critical_snaps = []
for step in range(1, 50001):
    i, j = rng.integers(0, L, size=2)
    grid[i, j] += 1
    if grid[i, j] >= Zc:
        relax(rng, grid)
    if step >= 20000 and (step - 20000) % 100 == 0:
        critical_snaps.append(grid.copy())

print(f"暂态快照数: {len(transient_snaps)}, 临界快照数: {len(critical_snaps)}")
print(f"暂态平均高度: {np.mean([g.mean() for g in transient_snaps]):.3f}")
print(f"临界平均高度: {np.mean([g.mean() for g in critical_snaps]):.3f}")

import numpy as np
from scipy import stats

L, Zc = 20, 4

def relax(rng, grid):
    """对给定网格执行异步弛豫至稳定"""
    # 找出所有不稳定格（简化入口，直接全扫描）
    while True:
        unst = np.argwhere(grid >= Zc)
        if len(unst) == 0:
            break
        k = rng.integers(0, len(unst))
        i0, j0 = unst[k]
        grid[i0, j0] -= Zc
        for di, dj in ((0,1),(0,-1),(1,0),(-1,0)):
            ni, nj = i0+di, j0+dj
            if 0 <= ni < L and 0 <= nj < L:
                grid[ni, nj] += 1

def evolve(L, Zc, total_steps, seed):
    rng = np.random.default_rng(seed)
    grid = np.zeros((L, L), dtype=int)
    for _ in range(total_steps):
        i, j = rng.integers(0, L, size=2)
        grid[i, j] += 1
        if grid[i, j] >= Zc:
            relax(rng, grid)
    return grid

# 暂态系综：200次独立短演化，各取第1000步快照
transient_snaps = []
for s in range(200):
    transient_snaps.append(evolve(L, Zc, 1000, seed=1000 + s))

# 临界系综：1次长演化，每100步存快照
rng = np.random.default_rng(42)
grid = np.zeros((L, L), dtype=int)
critical_snaps = []
for step in range(1, 50001):
    i, j = rng.integers(0, L, size=2)
    grid[i, j] += 1
    if grid[i, j] >= Zc:
        relax(rng, grid)
    if step >= 20000 and (step - 20000) % 100 == 0:
        critical_snaps.append(grid.copy())

print(f"暂态快照数: {len(transient_snaps)}, 临界快照数: {len(critical_snaps)}")
print(f"暂态平均高度: {np.mean([g.mean() for g in transient_snaps]):.3f}")
print(f"临界平均高度: {np.mean([g.mean() for g in critical_snaps]):.3f}")

def stimulate(snapshot, strength=2, seed=0):
    rng = np.random.default_rng(seed)
    grid = snapshot.copy()
    c = L // 2
    grid[c, c] += strength
    size, depth = 0, -1
    if grid[c, c] >= Zc:
        stack = [(c, c, 0)]
        while stack:
            k = rng.integers(0, len(stack))
            x, y, d = stack[k]; stack[k] = stack[-1]; stack.pop()
            if grid[x, y] < Zc:
                continue
            grid[x, y] -= Zc
            size += 1
            depth = max(depth, d)
            if grid[x, y] >= Zc:
                stack.append((x, y, d))
            for di, dj in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = x+di, y+dj
                if 0 <= nx < L and 0 <= ny < L:
                    grid[nx, ny] += 1
                    if grid[nx, ny] == Zc:
                        stack.append((nx, ny, d+1))
    return size, (depth + 1 if size > 0 else 0)

def summarize(name, snaps):
    sizes = np.array([stimulate(s, seed=k)[0] for k, s in enumerate(snaps)])
    depths = np.array([stimulate(s, seed=k)[1] for k, s in enumerate(snaps)])
    nz = sizes > 0
    print(f"{name}: 非零比例 {nz.mean()*100:.1f}%, 平均规模 {sizes.mean():.2f}, "
          f"最大 {sizes.max()}, P90={np.percentile(sizes,90):.0f}, "
          f"P99={np.percentile(sizes,99):.0f}, 平均深度 {depths[nz].mean():.2f}")
    return sizes, depths

s_sub, d_sub = summarize("暂态", transient_snaps)
s_crit, d_crit = summarize("临界", critical_snaps)
t, p = stats.ttest_ind(s_sub, s_crit, equal_var=False)
print(f"平均规模 Welch t检验: t={t:.2f}, p={p:.3f}")