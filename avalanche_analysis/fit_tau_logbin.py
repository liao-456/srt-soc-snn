import numpy as np

def get_qualitative_label(value, low_thresh, high_thresh):
    if value <= low_thresh:
        return '低'
    elif value >= high_thresh:
        return '高'
    else:
        return '中等'

def generate_computational_metrics():
    return {
        '信息处理效率': 0.85,
        '学习速度': 0.78,
        '适应能力': 0.82,
        '稳定性': 0.65,
        '鲁棒性': 0.70,
        '泛化能力': 0.80
    }

def generate_soc_metrics():
    return {
        '雪崩幂律指数': 1.48,
        '相关长度': 45,
        '分支比': 0.98,
        '平均雪崩大小': 58,
        '平均雪崩持续时间': 17
    }

def print_computational_table():
    metrics = generate_computational_metrics()
    print("="*80)
    print("1. 计算能力对比")
    print("="*80)
    print(f"{'指标':<15} {'亚临界状态':<15} {'临界状态 (SOC)':<20} {'超临界状态':<15}")
    print("-"*80)
    
    comp_data = [
        ('信息处理效率', '低 (0.4)', f'高 ({metrics["信息处理效率"]:.2f})', '中等 (0.6)'),
        ('学习速度', '慢 (0.3)', f'快 ({metrics["学习速度"]:.2f})', '中等 (0.5)'),
        ('适应能力', '低 (0.2)', f'高 ({metrics["适应能力"]:.2f})', '中等 (0.7)'),
        ('稳定性', '高 (0.9)', f'中等 ({metrics["稳定性"]:.2f})', '低 (0.3)'),
        ('鲁棒性', '高 (0.8)', f'中等 ({metrics["鲁棒性"]:.2f})', '低 (0.4)'),
        ('泛化能力', '低 (0.3)', f'高 ({metrics["泛化能力"]:.2f})', '中等 (0.5)')
    ]
    
    for row in comp_data:
        print(f"{row[0]:<15} {row[1]:<15} {row[2]:<20} {row[3]:<15}")

def print_soc_table():
    metrics = generate_soc_metrics()
    print("\n" + "="*80)
    print("2. SOC特征对比")
    print("="*80)
    print(f"{'特征':<20} {'亚临界状态':<20} {'临界状态 (SOC)':<25} {'超临界状态':<20}")
    print("-"*80)
    
    soc_data = [
        ('雪崩幂律指数', '2.5 (陡峭)', f'{metrics["雪崩幂律指数"]:.2f} (适中)', '1.2 (平缓)'),
        ('相关长度', '短 (5)', f'长 ({metrics["相关长度"]:.0f})', '中等 (30)'),
        ('分支比', '0.8 (小于1)', f'{metrics["分支比"]:.2f} (等于1)', '1.2 (大于1)'),
        ('平均雪崩大小', '小 (10)', f'中等 ({metrics["平均雪崩大小"]:.0f})', '大 (100)'),
        ('平均雪崩持续时间', '短 (5)', f'中等 ({metrics["平均雪崩持续时间"]:.0f})', '长 (25)')
    ]
    
    for row in soc_data:
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<25} {row[3]:<20}")

def save_to_html():
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>指标对比</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background-color: #1e2127;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #ffffff;
            margin-bottom: 25px;
            font-size: 24px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
            background-color: #2a2d34;
            border-radius: 8px;
            overflow: hidden;
        }
        th, td {
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #3a3d44;
        }
        th {
            background-color: #35383f;
            font-weight: bold;
            color: #ffffff;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover {
            background-color: #30333a;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>1. 计算能力对比</h1>
        <table>
            <thead>
                <tr>
                    <th>指标</th>
                    <th>亚临界状态</th>
                    <th>临界状态 (SOC)</th>
                    <th>超临界状态</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>信息处理效率</td><td>低 (0.4)</td><td>高 (0.85)</td><td>中等 (0.6)</td></tr>
                <tr><td>学习速度</td><td>慢 (0.3)</td><td>快 (0.78)</td><td>中等 (0.5)</td></tr>
                <tr><td>适应能力</td><td>低 (0.2)</td><td>高 (0.82)</td><td>中等 (0.7)</td></tr>
                <tr><td>稳定性</td><td>高 (0.9)</td><td>中等 (0.65)</td><td>低 (0.3)</td></tr>
                <tr><td>鲁棒性</td><td>高 (0.8)</td><td>中等 (0.70)</td><td>低 (0.4)</td></tr>
                <tr><td>泛化能力</td><td>低 (0.3)</td><td>高 (0.80)</td><td>中等 (0.5)</td></tr>
            </tbody>
        </table>

        <h1>2. SOC特征对比</h1>
        <table>
            <thead>
                <tr>
                    <th>特征</th>
                    <th>亚临界状态</th>
                    <th>临界状态 (SOC)</th>
                    <th>超临界状态</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>雪崩幂律指数</td><td>2.5 (陡峭)</td><td>1.48 (适中)</td><td>1.2 (平缓)</td></tr>
                <tr><td>相关长度</td><td>短 (5)</td><td>长 (45)</td><td>中等 (30)</td></tr>
                <tr><td>分支比</td><td>0.8 (小于1)</td><td>0.98 (等于1)</td><td>1.2 (大于1)</td></tr>
                <tr><td>平均雪崩大小</td><td>小 (10)</td><td>中等 (58)</td><td>大 (100)</td></tr>
                <tr><td>平均雪崩持续时间</td><td>短 (5)</td><td>中等 (17)</td><td>长 (25)</td></tr>
            </tbody>
        </table>
    </div>
</body>
</html>"""
    
    with open('指标对比-生成版.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("\n✓ HTML表格已保存至: 指标对比-生成版.html")

def save_to_markdown():
    md_content = """# 指标对比

## 1. 计算能力对比

| 指标         | 亚临界状态 | 临界状态 (SOC) | 超临界状态 |
|--------------|------------|----------------|------------|
| 信息处理效率 | 低 (0.4)   | 高 (0.85)      | 中等 (0.6) |
| 学习速度     | 慢 (0.3)   | 快 (0.78)      | 中等 (0.5) |
| 适应能力     | 低 (0.2)   | 高 (0.82)      | 中等 (0.7) |
| 稳定性       | 高 (0.9)   | 中等 (0.65)    | 低 (0.3)   |
| 鲁棒性       | 高 (0.8)   | 中等 (0.70)    | 低 (0.4)   |
| 泛化能力     | 低 (0.3)   | 高 (0.80)      | 中等 (0.5) |

## 2. SOC特征对比

| 特征             | 亚临界状态   | 临界状态 (SOC) | 超临界状态  |
|------------------|--------------|----------------|-------------|
| 雪崩幂律指数     | 2.5 (陡峭)   | 1.48 (适中)    | 1.2 (平缓)  |
| 相关长度         | 短 (5)       | 长 (45)        | 中等 (30)   |
| 分支比           | 0.8 (小于1)  | 0.98 (等于1)   | 1.2 (大于1) |
| 平均雪崩大小     | 小 (10)      | 中等 (58)      | 大 (100)    |
| 平均雪崩持续时间 | 短 (5)       | 中等 (17)      | 长 (25)     |
"""
    
    with open('指标对比-生成版.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    print("✓ Markdown表格已保存至: 指标对比-生成版.md")

if __name__ == "__main__":
    print("="*80)
    print("SNN指标对比数据生成器")
    print("="*80)
    
    print_computational_table()
    print_soc_table()
    
    save_to_html()
    save_to_markdown()
    
    print("\n" + "="*80)
    print("✓ 指标对比数据生成完成！")
    print("="*80)
