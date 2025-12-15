import os
# 设置环境变量，允许 OpenMP 库重复加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 超参数设置 (Hyperparameters)
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20
LATENT_DIM = 2  # 设置为2是为了方便画出平面“流形”图
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 2. 数据准备 (Data Loading)
# ==========================================
# 使用 MNIST 手写数字数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. 模型定义 (Model Architecture)
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        """
        对应论文章节：3. 数学模型
        """
        super(VAE, self).__init__()

        # --- 编码器 (Encoder): q_phi(z|x) ---
        # 作用：将高维输入 x 压缩为潜在分布的参数 mu 和 log_var
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 预测均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 预测对数方差

        # --- 解码器 (Decoder): p_theta(x|z) ---
        # 作用：将潜在变量 z 还原为数据 x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出像素值在 0-1 之间
        )

    def reparameterize(self, mu, logvar):
        """
        对应论文章节：3.4 重参数化技巧 (The Reparameterization Trick)
        公式: z = mu + sigma * epsilon
        统计意义：将随机性转移到 epsilon，使得 mu 和 sigma 可导
        """
        if self.training:
            std = torch.exp(0.5 * logvar) # log_var -> std
            eps = torch.randn_like(std)   # 从标准正态分布 N(0, I) 采样 epsilon
            return mu + std * eps
        else:
            return mu  # 测试时直接使用均值，保证结果确定性

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        z = self.reparameterize(mu, logvar)
        
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# ==========================================
# 4. 损失函数 (ELBO Loss)
# ==========================================
def loss_function(recon_x, x, mu, logvar):
    """
    对应论文章节：3.2 证据下界 (ELBO)
    Loss = Reconstruction_Loss + KL_Divergence
    """
    # 1. 重构损失 (Reconstruction Loss): 这里的负对数似然等价于 BCE
    # 也就是希望生成的图片和原图越像越好
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 2. KL 散度 (KL Divergence): D_KL(q(z|x) || p(z))
    # 公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 统计意义：强迫后验分布接近标准正态分布 N(0, I)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# ==========================================
# 5. 训练循环 (Training Loop)
# ==========================================
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []

print("开始训练...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        
        # 前向传播
        recon_batch, mu, logvar = model(data)
        
        # 计算损失
        loss = loss_function(recon_batch, data, mu, logvar)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    print(f'Epoch: {epoch+1}/{EPOCHS}, Average Loss (Negative ELBO): {avg_loss:.4f}')

torch.save(model.state_dict(), f'vae_mnist_dim{LATENT_DIM}.pth')

print(f"模型参数已保存至 vae_mnist_dim{LATENT_DIM}.pth")

# ==========================================
# 6. 结果可视化 (Visualization for Paper)
# ==========================================
model.eval()

# --- 图1: 损失收敛曲线 ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('ELBO Loss Convergence')
plt.xlabel('Epochs')
plt.ylabel('Negative ELBO')
plt.legend()
plt.grid(True)
plt.savefig('vae_loss.png') # 保存用于论文
plt.show()

# --- 图2: 2D 潜在空间流形分布 (Latent Manifold) ---
# 仅当 latent_dim=2 时有效
if LATENT_DIM == 2:
    print("正在生成潜在空间分布图...")
    plt.figure(figsize=(10, 8))
    
    # 抽取测试集中的样本点
    test_batch_size = 5000
    images, labels = next(iter(DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)))
    images = images.to(DEVICE)
    
    with torch.no_grad():
        _, mu, _ = model(images)
        z = mu.cpu().numpy() # 取均值作为坐标
        
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar()
    plt.title('2D Latent Space Manifold (Color=Digit Class)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)
    plt.savefig('vae_latent_manifold.png') # 保存用于论文
    plt.show()

# --- 图3: 潜在空间插值生成 (Manifold Traversing) ---
# 在潜在空间网格采样，看看能生成什么图
if LATENT_DIM == 2:
    print("正在生成数字流形图...")
    n = 20  # 20x20 的网格
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # 在 [-3, 3] 的区间内线性采样，对应正态分布 99.7% 的区间
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1] # 翻转Y轴以符合直角坐标系
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(DEVICE)
            with torch.no_grad():
                x_decoded = model.decoder(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size).cpu().numpy()
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
            
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Generated Digits from Latent Manifold')
    plt.axis('off')
    plt.savefig('vae_generated_manifold.png') # 保存用于论文
    plt.show()

print("完成！图片已保存。")