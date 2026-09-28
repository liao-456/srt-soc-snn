import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# ========== 崩塌函数（通用） ==========
def topple(sandpile, Zc=4):
    new_pile = sandpile.copy()
    avalanche_happened = False
    avalanche_size = 0
    for i in range(sandpile.shape[0]):
        for j in range(sandpile.shape[1]):
            if sandpile[i, j] >= Zc:
                avalanche_happened = True
                avalanche_size += 1
                new_pile[i, j] -= Zc
                if i > 0: new_pile[i-1, j] += 1
                if i < sandpile.shape[0]-1: new_pile[i+1, j] += 1
                if j > 0: new_pile[i, j-1] += 1
                if j < sandpile.shape[1]-1: new_pile[i, j+1] += 1
    return new_pile, avalanche_happened, avalanche_size

# ========== 单次模拟函数 ==========
def run_simulation(L, Zc=4, total_steps=10000, seed=42):
    np.random.seed(seed)
    sandpile = np.zeros((L, L), dtype=int)
    avalanche_sizes = []  # 记录每次雪崩规模（>0的才记录）
    
    for step in range(total_steps):
        i, j = np.random.randint(0, L, size=2)
        sandpile[i, j] += 1
        
        current_size = 0
        while True:
            sandpile, happened, size_step = topple(sandpile, Zc)
            if happened:
                current_size += size_step
            else:
                break
        if current_size > 0:
            avalanche_sizes.append(current_size)
    return avalanche_sizes

# ========== 熵变分析 ==========
def calculate_entropy(sizes_list):
    if len(sizes_list) == 0:
        return 0.0
    values, counts = np.unique(sizes_list, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))

def entropy_evolution(avalanche_records, total_steps, window_size=500):
    """avalanche_records: list of (step, size)"""
    times, entropies = [], []
    for start in range(0, total_steps, window_size):
        end = start + window_size
        sizes_in_window = [size for (s, size) in avalanche_records if start <= s < end]
        entropies.append(calculate_entropy(sizes_in_window))
        times.append(start)
    return times, entropies

# 实验参数
sizes = [20, 30, 50]     # 网格尺寸
Zc = 4
total_steps = 10000       # 如果50x50太慢，可临时改为5000看趋势
window_size = 500

results = {}  # 存储每个尺寸的结果
all_avalanche_records = {}  # 用于熵计算的带时间戳记录

for L in sizes:
    print(f"\n开始模拟 L={L}×{L}，预计耗时...")
    t0 = time.time()
    
    # 运行模拟（这里保存详细的(step, size)以便做熵分析）
    np.random.seed(42)
    sandpile = np.zeros((L, L), dtype=int)
    av_records = []  # (step, size)
    av_sizes = []    # 只存size
    
    for step in range(total_steps):
        i, j = np.random.randint(0, L, size=2)
        sandpile[i, j] += 1
        
        current_size = 0
        while True:
            sandpile, happened, size_step = topple(sandpile, Zc)
            if happened:
                current_size += size_step
            else:
                break
        if current_size > 0:
            av_sizes.append(current_size)
            av_records.append((step, current_size))
    
    t1 = time.time()
    print(f"L={L} 完成，耗时 {t1-t0:.1f} 秒，雪崩次数 {len(av_sizes)}")
    
    results[L] = {
        'sizes': av_sizes,
        'max': max(av_sizes) if av_sizes else 0,
        'mean': np.mean(av_sizes) if av_sizes else 0,
        'count': len(av_sizes)
    }
    all_avalanche_records[L] = av_records

print("\n全部模拟完成！")

plt.figure(figsize=(12,6))
colors = {20: 'b', 30: 'g', 50: 'r'}
for L in sizes:
    times, ent = entropy_evolution(all_avalanche_records[L], total_steps, window_size)
    plt.plot(times, ent, color=colors[L], linewidth=2, marker='o', markersize=3, label=f'{L}×{L}')
plt.xlabel('模拟时间步')
plt.ylabel('香农熵 (H)')
plt.title('不同网格尺寸下熵变曲线对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('entropy_compare_multisize.png', dpi=300, bbox_inches='tight')
plt.show()

def fit_powerlaw_pdf(sizes, L, plot=True):
    """使用对数分箱的PDF拟合，返回tau和R^2"""
    max_s = np.max(sizes)
    # 对数间隔bins，从1到max_s
    bins = np.logspace(np.log10(1), np.log10(max_s), 30)
    counts, edges = np.histogram(sizes, bins=bins)
    # 每个bin的几何中心
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    # 概率密度
    pdf = counts / (len(sizes) * widths)
    
    # 过滤非零
    nonzero = counts > 0
    x_data = centers[nonzero]
    y_data = pdf[nonzero]
    
    # 拟合范围：避开太小和太大
    fit_mask = (x_data >= 5) & (x_data <= max_s*0.5)
    if np.sum(fit_mask) < 5:
        print(f"  L={L}: 拟合数据点不足，跳过")
        return None, None
    
    logx = np.log(x_data[fit_mask])
    logy = np.log(y_data[fit_mask])
    slope, intercept, rval, pval, stderr = linregress(logx, logy)
    tau = -slope  # PDF ∝ s^{-tau}
    r2 = rval**2
    
    if plot:
        plt.figure(figsize=(7,5))
        plt.loglog(x_data, y_data, 'bo', markersize=3, label='PDF数据')
        x_fit = x_data[fit_mask]
        y_fit = np.exp(intercept) * x_fit**(-tau)
        plt.loglog(x_fit, y_fit, 'r-', linewidth=2,
           label=r'$\tau$=' + f'{tau:.3f}' + '\n' + r'$R^2$=' + f'{r2:.3f}')
        plt.xlabel('雪崩规模 s')
        plt.ylabel('P(s)')
        plt.title(f'PDF幂律拟合 (L={L})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(f'pdf_fit_L{L}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return tau, r2

def fit_powerlaw_ccdf(sizes, L, plot=True):
    """CCDF拟合，返回alpha和R^2（注意这里alpha是CCDF指数，τ=1-alpha）"""
    sorted_sizes = np.sort(sizes)
    n = len(sorted_sizes)
    ccd = 1.0 - np.arange(1, n+1)/n
    # 剔除重复值影响，取唯一值
    # 简单方法：直接使用所有点
    # 拟合范围
    max_s = np.max(sizes)
    mask = (sorted_sizes >= 5) & (sorted_sizes <= max_s*0.5)
    if np.sum(mask) < 5:
        print(f"  L={L}: CCDF拟合数据点不足")
        return None, None
    x_fit = sorted_sizes[mask]
    y_fit = ccd[mask]
    logx = np.log(x_fit)
    logy = np.log(y_fit)
    slope, intercept, rval, pval, stderr = linregress(logx, logy)
    alpha_ccdf = -slope  # CCDF ∝ s^{-alpha}
    r2 = rval**2
    tau_from_ccdf = 1 + alpha_ccdf  # 因为 CCDF指数 = τ - 1
    
    if plot:
        plt.figure(figsize=(7,5))
        plt.loglog(sorted_sizes, ccd, 'bo', markersize=3, label='CCDF数据')
        y_fit_line = np.exp(intercept) * x_fit**(-alpha_ccdf)
        plt.loglog(x_fit, y_fit_line, 'r-', linewidth=2,
           label=r'$\alpha$=' + f'{alpha_ccdf:.3f}' + r' ($\tau \approx$' + f'{tau_from_ccdf:.3f})')
        plt.xlabel('雪崩规模 s')
        plt.ylabel('P(S ≥ s)')
        plt.title(f'CCDF幂律拟合 (L={L})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(f'ccdf_fit_L{L}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return alpha_ccdf, r2, tau_from_ccdf

summary = []  # 汇总表

for L in sizes:
    av_sizes = results[L]['sizes']
    print(f"\n====== L={L}×{L} ======")
    # PDF拟合
    tau, r2_pdf = fit_powerlaw_pdf(av_sizes, L, plot=True)
    # CCDF拟合
    alpha_ccdf, r2_ccdf, tau_ccdf = fit_powerlaw_ccdf(av_sizes, L, plot=True)
    
    summary.append({
        'L': L,
        '雪崩次数': results[L]['count'],
        '平均规模': results[L]['mean'],
        '最大规模': results[L]['max'],
        'τ (PDF)': tau if tau else 'N/A',
        'R² (PDF)': r2_pdf if r2_pdf else 'N/A',
        'α (CCDF)': alpha_ccdf if alpha_ccdf else 'N/A',
        'τ (CCDF推算)': tau_ccdf if tau_ccdf else 'N/A'
    })

# 打印汇总表格
print("\n========== 多尺寸对比汇总 ==========")
print("{:<8} {:<12} {:<10} {:<10} {:<12} {:<12} {:<12} {:<15}".format(
    '尺寸','雪崩次数','平均规模','最大规模','τ(PDF)','R²(PDF)','α(CCDF)','τ(CCDF推算)'))
for row in summary:
    print("{:<8} {:<12} {:<10.2f} {:<10} {:<12} {:<12} {:<12} {:<15}".format(
        f"{row['L']}×{row['L']}",
        row['雪崩次数'],
        row['平均规模'],
        row['最大规模'],
        f"{row['τ (PDF)']:.3f}" if isinstance(row['τ (PDF)'], float) else row['τ (PDF)'],
        f"{row['R² (PDF)']:.3f}" if isinstance(row['R² (PDF)'], float) else row['R² (PDF)'],
        f"{row['α (CCDF)']:.3f}" if isinstance(row['α (CCDF)'], float) else row['α (CCDF)'],
        f"{row['τ (CCDF推算)']:.3f}" if isinstance(row['τ (CCDF推算)'], float) else row['τ (CCDF推算)']
    ))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 图1：三种尺寸的PDF幂律拟合 ====================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for idx, L in enumerate([20, 30, 50]):
    ax = axes[idx]
    sizes = results[L]['sizes']
    
    # 对数分箱
    max_s = np.max(sizes)
    bins = np.logspace(np.log10(1), np.log10(max_s), 30)
    counts, edges = np.histogram(sizes, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    pdf = counts / (len(sizes) * widths)
    
    nonzero = counts > 0
    x_data = centers[nonzero]
    y_data = pdf[nonzero]
    
    # 拟合
    fit_mask = (x_data >= 5) & (x_data <= max_s * 0.5)
    logx = np.log(x_data[fit_mask])
    logy = np.log(y_data[fit_mask])
    slope, intercept, rval, pval, stderr = linregress(logx, logy)
    tau = -slope
    r2 = rval**2
    
    # 画图
    ax.loglog(x_data, y_data, 'bo', markersize=3)
    x_fit = x_data[fit_mask]
    y_fit = np.exp(intercept) * x_fit**(-tau)
    ax.loglog(x_fit, y_fit, 'r-', linewidth=2,
          label=r'$\tau$=' + f'{tau:.3f}' + '\n' + r'$R^2$=' + f'{r2:.3f}')
    ax.set_xlabel('雪崩规模 s')
    ax.set_ylabel('P(s)')
    ax.set_title(f'L={L}')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig1_powerlaw_multisize.png', dpi=300, bbox_inches='tight')
plt.show()