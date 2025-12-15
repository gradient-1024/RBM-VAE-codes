import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# 0. 配置与环境准备
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"图片将保存在: ./{RESULTS_DIR}/ 目录下")

# 1. 配置参数
BATCH_SIZE = 64
VISIBLE_UNITS = 784 
HIDDEN_UNITS = 256   
K_STEPS = 1          
EPOCHS = 60          
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {DEVICE}")

# 2. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x)) 
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. RBM 模型定义
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k=1):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01) 
        self.v_bias = nn.Parameter(torch.zeros(n_vis))          
        self.h_bias = nn.Parameter(torch.zeros(n_hid))          
        self.k = k  

    def sample_h(self, v):
        # 给定可见层 v，采样隐藏层 h
        activation = torch.mm(v, self.W) + self.h_bias
        p_h = torch.sigmoid(activation)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    def sample_v(self, h):
        # 给定隐藏层 h，采样可见层 v
        activation = torch.mm(h, self.W.t()) + self.v_bias
        p_v = torch.sigmoid(activation)
        v_sample = torch.bernoulli(p_v)
        return p_v, v_sample

    def forward(self, v):
        # 从 v 得到重构的 v_prime
        _, h = self.sample_h(v)
        p_v, _ = self.sample_v(h)
        return p_v

    def contrastive_divergence(self, v0):
        # 正相
        p_h0, h0 = self.sample_h(v0)
        positive_grad = torch.mm(v0.t(), p_h0)

        # 负相
        vk = v0
        hk = h0
        for _ in range(self.k):
            _, vk = self.sample_v(hk) 
            p_hk, hk = self.sample_h(vk)
        
        negative_grad = torch.mm(vk.t(), p_hk)

        batch_size = v0.size(0)
        
        self.W.data += LEARNING_RATE * (positive_grad - negative_grad) / batch_size
        self.v_bias.data += LEARNING_RATE * torch.mean(v0 - vk, dim=0)
        self.h_bias.data += LEARNING_RATE * torch.mean(p_h0 - p_hk, dim=0)

        return torch.mean((v0 - vk)**2)

# 4. 训练循环
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, K_STEPS).to(DEVICE)

print("Starting Training...")
loss_history = []

for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        data = (data > 0.5).float()
        loss = rbm.contrastive_divergence(data)
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Reconstruction Error: {avg_loss:.4f}")

torch.save(rbm.state_dict(), 'RBM_mnist.pth')
print(f"模型参数已保存至 RBM_mnist.pth")


plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.title("RBM Training Reconstruction Error")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
loss_plot_path = os.path.join(RESULTS_DIR, "training_loss.png")
plt.savefig(loss_plot_path)
plt.close() 
print(f"训练损失曲线已保存至: {loss_plot_path}")


# 5. 结果可视化
def show_and_save_images(images, title, filename):
    images = images.view(-1, 1, 28, 28).cpu().detach()
    grid_img = make_grid(images, nrow=8, padding=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"已保存图像: {save_path}")


# 任务 A: 图像重构 (Reconstruction)
test_images = next(iter(test_loader))[0][:32].to(DEVICE)
test_images_bin = (test_images > 0.5).float() 
reconstructed_probs = rbm(test_images_bin) 


show_and_save_images(test_images_bin, "Original Images (Binarized)", "original_images.png")
show_and_save_images(reconstructed_probs, "Reconstructed Images (RBM Output)", "reconstructed_images.png")

# 任务 B: 图像生成 (Generation / Dreaming)
print("Generating new images from noise (Gibbs Sampling)...")

noise = torch.rand(32, VISIBLE_UNITS).to(DEVICE) 

curr_v = (noise > 0.5).float()
for step in range(1000):
    _, h = rbm.sample_h(curr_v)
    _, curr_v = rbm.sample_v(h)
    
    if step == 999:
        probs_v, _ = rbm.sample_v(h)
        curr_v = probs_v

show_and_save_images(curr_v, "Generated Images (Dreaming from Noise)", "generated_images_gibbs.png")

print("图像生成保存完毕")