import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体，解决图表中的中文显示问题
my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams['axes.unicode_minus'] = False

# ==================== 基础MLP模型 ====================
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 200),   # 输入层
            nn.ReLU(),
            nn.Linear(200, 200),   # 中间层
            nn.ReLU(),
            nn.Linear(200, 100)    # 输出层
        )
    
    def forward(self, x):
        return self.model(x)

# ==================== 生成合成时间序列数据 ====================
def make_data():
    # 生成初始状态
    current = torch.randn(500, 100) * 0.5
    # 模拟下一帧数据（引入运动量及噪声）
    next_frame = current + 0.1 + torch.randn(500, 100) * 0.05
    return current, next_frame

# ==================== 模型训练 ====================
def train():
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    X, y = make_data()
    
    print("开始训练无记忆模型...")
    for epoch in range(40):           # 训练40轮
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"第 {epoch} 轮，误差 = {loss.item():.4f}")
    
    print("训练完成。")
    return model, X, y

# ==================== 结果可视化 ====================
def show_result(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X[0:5])
    
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(X[i].numpy(), label='当前', alpha=0.7)
        plt.plot(y[i].numpy(), label='真实下一帧', alpha=0.7)
        plt.plot(pred[i].numpy(), label='模型预测', linestyle='--')
        plt.legend(prop=my_font)
    
    plt.suptitle("MLP预测效果（无记忆模型）", fontproperties=my_font)
    
    # 保存图片
    plt.savefig('plain_mlp_result.png', dpi=300)
    print("图表已保存为 plain_mlp_result.png")

# ==================== 主程序运行 ====================
if __name__ == "__main__":
    model, X, y = train()
    show_result(model, X, y)
    print("程序运行结束。")