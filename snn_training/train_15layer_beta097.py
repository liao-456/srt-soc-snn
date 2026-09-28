import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate, functional, utils
import matplotlib.pyplot as plt
import time

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support.")
device = torch.device("cuda")
print(f"Using device: {device}")

batch_size = 128
num_steps = 30
lr = 1e-3
beta = 0.97
spike_grad = surrogate.fast_sigmoid()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, beta=0.97, spike_grad=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x, mem1, mem2):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out, mem1 = self.lif1(out, mem1)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out, mem2 = self.lif2(out, mem2)
        return out, mem1, mem2

class DeepSNN15(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = beta
        self.spike_grad = spike_grad
        
        self.init_conv = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.init_lif = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(32, 32, stride=1, beta=self.beta, spike_grad=self.spike_grad),
            ResidualBlock(32, 32, stride=1, beta=self.beta, spike_grad=self.spike_grad),
            ResidualBlock(32, 64, stride=2, beta=self.beta, spike_grad=self.spike_grad),
            ResidualBlock(64, 64, stride=1, beta=self.beta, spike_grad=self.spike_grad),
            ResidualBlock(64, 128, stride=2, beta=self.beta, spike_grad=self.spike_grad),
            ResidualBlock(128, 128, stride=1, beta=self.beta, spike_grad=self.spike_grad),
            ResidualBlock(128, 256, stride=2, beta=self.beta, spike_grad=self.spike_grad),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10, bias=False)
        self.lif_out = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, output=True)
    
    def forward(self, x):
        mem_init = self.init_lif.init_leaky()
        block_mems = []
        for blk in self.blocks:
            block_mems.append([blk.lif1.init_leaky(), blk.lif2.init_leaky()])
        mem_out = self.lif_out.init_leaky()
        
        spk_rec = []
        for step in range(num_steps):
            out = self.init_conv(x)
            out = self.init_bn(out)
            out, mem_init = self.init_lif(out, mem_init)
            
            for idx, blk in enumerate(self.blocks):
                out, block_mems[idx][0], block_mems[idx][1] = blk(out, block_mems[idx][0], block_mems[idx][1])
            
            out = self.global_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out, mem_out = self.lif_out(out, mem_out)
            spk_rec.append(out)
        return torch.stack(spk_rec, dim=0)

net = DeepSNN15().to(device)
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params:,}")

loss_fn = functional.ce_rate_loss()
optimizer = optim.Adam(net.parameters(), lr=lr)

epoch_times = []
test_times = []
epoch_accuracies = []
epoch_spike_rates = []

def test():
    net.eval()
    start_time = time.time()
    correct = 0
    total = 0
    total_spikes = 0
    total_neurons = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            utils.reset(net)
            spk_rec = net(data)
            total_spikes += spk_rec.sum().item()
            total_neurons += spk_rec.numel()
            spike_counts = spk_rec.sum(dim=0)
            pred = spike_counts.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    accuracy = 100 * correct / total
    spike_rate = total_spikes / total_neurons
    inference_time = time.time() - start_time
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Spike Rate (per neuron per time step): {spike_rate:.4f}')
    print(f'Test Inference Time: {inference_time:.2f}s')
    return accuracy, spike_rate, inference_time

def train():
    net.train()
    total_train_start = time.time()
    for epoch in range(1, 11):
        epoch_start = time.time()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            utils.reset(net)
            spk_rec = net(data)
            loss = loss_fn(spk_rec, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')
        acc, spike_rate, test_time = test()
        test_times.append(test_time)
        epoch_accuracies.append(acc)
        epoch_spike_rates.append(spike_rate)
    total_train_time = time.time() - total_train_start

    print("\n" + "="*60)
    print("              SNN SoC Analysis Report")
    print("="*60)
    print(f"Model Architecture        : 15-layer Residual SNN (1 initial conv 5x5 + 7 blocks × 2 conv)")
    print(f"Dataset                   : MNIST (28x28 grayscale, 10 classes)")
    print(f"Number of trainable params: {total_params:,}")
    print(f"Batch size                : {batch_size}")
    print(f"Time steps (num_steps)    : {num_steps}")
    print(f"Membrane decay (beta)     : {beta}")
    print(f"Learning rate             : {lr}")
    print(f"Optimizer                 : Adam")
    print(f"Spike gradient surrogate  : fast_sigmoid()")
    print("-"*60)
    print("Epoch | Train Time(s) | Test Time(s) | Accuracy(%) | Spike Rate")
    for i in range(len(epoch_times)):
        print(f"  {i+1}   |    {epoch_times[i]:.2f}      |    {test_times[i]:.2f}      |    {epoch_accuracies[i]:.2f}     |    {epoch_spike_rates[i]:.4f}")
    print("-"*60)
    print(f"Total training time       : {total_train_time:.2f}s")
    avg_acc = sum(epoch_accuracies)/len(epoch_accuracies)
    avg_spike = sum(epoch_spike_rates)/len(epoch_spike_rates)
    print(f"Average Accuracy          : {avg_acc:.2f}%")
    print(f"Average Spike Rate        : {avg_spike:.4f} (per neuron per time step)")
    print("="*60)
    print("Note: Spike Rate is the average number of spikes generated per output neuron per time step.")
    print("      Lower values indicate sparser activity, reducing compute and energy demands on SoC.")

if __name__ == "__main__":
    train()