import os
# 设置环境变量，允许 OpenMP 库重复加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 必须重新定义网络结构 (与训练时完全一致)
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2): # 注意 latent_dim 要对应
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# ==========================================
# 2. 加载模型
# ==========================================
# 【修改这里】
LATENT_DIM = 2  # 改为 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A. 实例化模型
# 这里会自动使用 LATENT_DIM=2 初始化网络结构，必须与训练时一致
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

# B. 加载参数文件
# 代码会自动寻找 'vae_mnist_dim2.pth'
# 请确保您的文件夹里有这个文件（即您之前跑过 dim=2 的训练）
checkpoint_path = f'vae_mnist_dim{LATENT_DIM}.pth' 

try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    print(f"成功加载模型: {checkpoint_path}")
except FileNotFoundError:
    print(f"错误: 找不到文件 {checkpoint_path}。请先运行训练脚本并设置 LATENT_DIM=2。")

# ==========================================
# 3. 直接使用：可视化
# ==========================================
import numpy as np

# 如果是 2维，画出炫酷的流形分布图
if LATENT_DIM == 2:
    print("检测到 Dim=2，正在绘制 2D 流形网格...")
    n = 20  # 20x20 的网格
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # 在 [-3, 3] 区间采样
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
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
    plt.title('2D Manifold from Loaded Model')
    plt.axis('off')
    plt.show()

# 如果是其他维度（比如 10），画随机生成图
else:
    print(f"检测到 Dim={LATENT_DIM}，随机生成一张图片...")
    with torch.no_grad():
        z_sample = torch.randn(1, LATENT_DIM).to(DEVICE) 
        generated_img = model.decoder(z_sample)
        img = generated_img.view(28, 28).cpu().numpy()

    plt.imshow(img, cmap='Greys_r')
    plt.title(f"Random Generation (Dim={LATENT_DIM})")
    plt.axis('off')
    plt.show()