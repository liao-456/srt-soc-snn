# -*- coding: utf-8 -*-
"""
perturbation_ensemble.py —— 快照系综扰动实验
产出：控制台打印暂态/临界两系的平均规模、P90/P99/最大值、Welch t检验
对应论文：4.1节末段
运行：python perturbation_ensemble.py
"""
import numpy as np
from scipy import stats

L, Zc = 20, 4


def relax(rng, grid):
    """对给定网格执行异步弛豫至稳定。"""
    while True:
        unst = np.argwhere(grid >= Zc)
        if len(unst) == 0:
            break
        k = rng.integers(0, len(unst))
        i0, j0 = unst[k]
        grid[i0, j0] -= Zc
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ni, nj = i0 + di, j0 + dj
            if 0 <= ni < L and 0 <= nj < L:
                grid[ni, nj] += 1


def evolve(total_steps, seed):
    """从平坦初态演化 total_steps 步，返回末态网格。"""
    rng = np.random.default_rng(seed)
    grid = np.zeros((L, L), dtype=int)
    for _ in range(total_steps):
        i, j = rng.integers(0, L, size=2)
        grid[i, j] += 1
        if grid[i, j] >= Zc:
            relax(rng, grid)
    return grid


def stimulate(snapshot, seed):
    """在中心格施加+2刺激，返回诱发雪崩的规模与级联深度。"""
    rng = np.random.default_rng(seed)
    grid = snapshot.copy()
    c = L // 2
    grid[c, c] += 2
    size, depth = 0, -1
    if grid[c, c] >= Zc:
        stack = [(c, c, 0)]
        while stack:
            k = rng.integers(0, len(stack))
            x, y, d = stack[k]
            stack[k] = stack[-1]
            stack.pop()
            if grid[x, y] < Zc:
                continue
            grid[x, y] -= Zc
            size += 1
            depth = max(depth, d)
            if grid[x, y] >= Zc:
                stack.append((x, y, d))
            for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + di, y + dj
                if 0 <= nx < L and 0 <= ny < L:
                    grid[nx, ny] += 1
                    if grid[nx, ny] == Zc:
                        stack.append((nx, ny, d + 1))
    return size, (depth + 1 if size > 0 else 0)


def summarize(name, snaps):
    sizes = np.array([stimulate(s, seed=k)[0] for k, s in enumerate(snaps)])
    depths = np.array([stimulate(s, seed=k)[1] for k, s in enumerate(snaps)])
    nz = sizes > 0
    print(f"{name}: 快照{len(snaps)}个, 非零比例{nz.mean()*100:.1f}%, "
          f"平均规模{sizes.mean():.2f}, 最大{sizes.max()}, "
          f"P90={np.percentile(sizes, 90):.0f}, P99={np.percentile(sizes, 99):.0f}, "
          f"平均深度{depths[nz].mean():.2f}")
    return sizes


if __name__ == "__main__":
    # 暂态系综：200次独立短演化，各取第1000步快照
    trans = [evolve(1000, seed=1000 + s) for s in range(200)]

    # 临界系综：1次长演化（50000步，seed=42），第20000步起每100步取一张
    rng = np.random.default_rng(42)
    grid = np.zeros((L, L), dtype=int)
    crit = []
    for step in range(1, 50001):
        i, j = rng.integers(0, L, size=2)
        grid[i, j] += 1
        if grid[i, j] >= Zc:
            relax(rng, grid)
        if step >= 20000 and (step - 20000) % 100 == 0:
            crit.append(grid.copy())

    print(f"暂态快照数: {len(trans)}, 临界快照数: {len(crit)}")
    print(f"暂态平均高度: {np.mean([g.mean() for g in trans]):.3f}")
    print(f"临界平均高度: {np.mean([g.mean() for g in crit]):.3f}")

    s1 = summarize("暂态", trans)
    s2 = summarize("临界", crit)

    t, p = stats.ttest_ind(s1, s2, equal_var=False)
    print(f"平均规模 Welch t检验: t={t:.2f}, p={p:.3f}")