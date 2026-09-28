import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体，解决图表中的中文显示问题
my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams['axes.unicode_minus'] = False

# ==================== 带记忆的MLP模型 ====================
class MemoryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 512)   # 输入为连续3帧拼接，维度为300
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 100)   # 输出下一帧，维度为100
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ==================== 生成时间序列数据 ====================
def make_time_data():
    num = 1000
    # 初始化序列状态
    data = torch.randn(num+3, 100) * 0.5
    # 生成具有时序依赖的序列数据
    for i in range(1, len(data)):
        data[i] = data[i-1] * 0.95 + torch.randn(100) * 0.08
    
    # 构造训练样本：拼接连续3帧作为输入，预测第4帧
    X = []
    y = []
    for i in range(len(data)-3):
        combined = torch.cat([data[i], data[i+1], data[i+2]])
        X.append(combined)
        y.append(data[i+3])
    
    return torch.stack(X), torch.stack(y)

# ==================== 模型训练 ====================
def train():
    model = MemoryMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    X, y = make_time_data()
    losses = []
    
    print("开始训练带记忆的模型...")
    for epoch in range(80):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"第 {epoch} 轮，误差 = {loss.item():.4f}")
    
    print("训练完成。")
    return model, X, y, losses

# ==================== 结果可视化 ====================
def show_result(model, X, y, losses):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("误差下降曲线", fontproperties=my_font)
    plt.xlabel("训练轮数", fontproperties=my_font)
    plt.ylabel("误差", fontproperties=my_font)
    
    # 绘制预测对比
    model.eval()
    with torch.no_grad():
        pred = model(X[0:5])
    
    plt.subplot(1, 2, 2)
    i = 0
    plt.plot(X[i][:100].numpy(), label='当前帧特征', alpha=0.6)
    plt.plot(y[i].numpy(), label='真实下一帧', alpha=0.8)
    plt.plot(pred[i].numpy(), label='模型预测', linestyle='--')
    plt.legend(prop=my_font)
    plt.title("带记忆的预测效果", fontproperties=my_font)
    
    # 保存图片
    plt.savefig('loss_curve.png', dpi=300)
    print("图表已保存为 loss_curve.png")

# ==================== 主程序运行 ====================
if __name__ == "__main__":
    model, X, y, losses = train()
    show_result(model, X, y, losses)
    print("程序运行结束。")