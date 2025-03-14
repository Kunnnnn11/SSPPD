import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 配置中文字体（例如使用 SimHei）和正常显示负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# -------------------------
# 改进后的正弦时间嵌入模块
# -------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """
    改进后的正弦时间嵌入模块。
    当 embedding_dim = 128 时，输出尺寸为 [B, 128]（包含 64 个 sin 与 64 个 cos 分量）。
    此模块预先计算频率向量并存入 register_buffer 中，确保每次 forward 返回正确的嵌入维度。
    """
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2  # 例如 128//2 = 64
        inv_freq = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / (half_dim - 1))
        # 注册 inv_freq 为缓冲变量，加载和保存时会自动处理
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        """
        参数 t：形状为 [B, 1] 或 [B]（内部会调整为 [B, 1]）
        返回：嵌入向量, 形状为 [B, embedding_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # 计算正弦和余弦嵌入
        sinusoid_inp = t * self.inv_freq.unsqueeze(0)
        sin_emb = torch.sin(sinusoid_inp)
        cos_emb = torch.cos(sinusoid_inp)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        return emb

# -------------------------
# 辅助函数：中心裁剪（用于 U-Net 拼接时对齐尺寸）
# -------------------------
def center_crop(tensor, target_h, target_w):
    """
    对 tensor 进行中心裁剪，使其高度为 target_h，宽度为 target_w
    """
    _, _, h, w = tensor.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return tensor[:, :, start_h: start_h + target_h, start_w: start_w + target_w]

# -------------------------
# 模型定义部分：残差块、下采样、上采样与 U-Net 架构
# -------------------------
class ResBlock(nn.Module):
    """
    残差块，采用卷积操作并将时间条件嵌入注入到卷积中。
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.relu(h)
        h = self.conv2(h)
        return self.relu(h + self.shortcut(x))

class DownSample(nn.Module):
    """
    下采样模块，利用卷积减少空间尺寸。
    """
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    """
    上采样模块，通过双线性插值恢复空间尺寸，并使用卷积进行特征提取。
    """
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)

class ScoreNetOptimized(nn.Module):
    """
    创新版打分网络，采用 U-Net 结构和残差块，
    同时利用正弦时间嵌入模块将时间条件注入网络中。
    默认参数：in_channels=1, base_channels=64, time_emb_dim=128。
    """
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super(ScoreNetOptimized, self).__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        # 下采样部分
        self.resblock1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.down1 = DownSample(base_channels)
        self.resblock2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownSample(base_channels * 2)
        self.resblock3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        # 上采样部分
        self.up1 = UpSample(base_channels * 4)
        self.resblock4 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = UpSample(base_channels * 2)
        self.resblock5 = ResBlock(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t.view(-1, 1))
        # 下采样部分
        x1 = self.resblock1(x, t_emb)  # [B, base_channels, H, W]
        x2 = self.down1(x1)            # [B, base_channels, H/2, W/2]
        x2 = self.resblock2(x2, t_emb)   # [B, base_channels*2, H/2, W/2]
        x3 = self.down2(x2)            # [B, base_channels*2, H/4, W/4]
        x3 = self.resblock3(x3, t_emb)   # [B, base_channels*4, H/4, W/4]
        # 上采样部分
        x_up = self.up1(x3)            # 期望输出尺寸：[B, base_channels*4, H/2, W/2]
        # 对 x_up 进行中心裁剪，使其与 x2 尺寸匹配
        _, _, h2, w2 = x2.shape
        x_up = center_crop(x_up, h2, w2)
        x_cat = torch.cat([x_up, x2], dim=1)  # 拼接后尺寸：[B, base_channels*4+ base_channels*2, H/2, W/2]
        x_mid = self.resblock4(x_cat, t_emb)   # [B, base_channels*2, H/2, W/2]
        x_up2 = self.up2(x_mid)          # 期望输出尺寸：[B, base_channels*2, H, W]
        # 对 x_up2 进行中心裁剪，使其与 x1 尺寸匹配
        _, _, h1, w1 = x1.shape
        x_up2 = center_crop(x_up2, h1, w1)
        x_cat2 = torch.cat([x_up2, x1], dim=1)  # 拼接后尺寸：[B, base_channels*2+ base_channels, H, W]
        x_out = self.resblock5(x_cat2, t_emb)  # [B, base_channels, H, W]
        score = self.final_conv(x_out)         # 输出形状：[B, in_channels, H, W]
        return score

# -------------------------
# 数据集定义
# -------------------------
class PrecipitationDataset(Dataset):
    """
    从指定目录中加载 .pt 文件，每个文件保存一个 [68,150] 的 tensor，
    转换为 [1,68,150] 的单通道图像，替换 nan 为 0 并归一化。
    """
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, "*.pt"))
        if not self.file_paths:
            raise ValueError("在指定目录中未找到 .pt 文件，请检查数据路径。")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sample = torch.load(file_path)
        if sample.dim() == 2:
            sample = sample.unsqueeze(0)
        sample[torch.isnan(sample)] = 0
        sample = (sample - sample.mean()) / (sample.std() + 1e-8)
        return sample

# -------------------------
# 辅助工具函数：添加噪声与结果可视化
# -------------------------
def add_noise(x, sigma):
    """
    向输入数据 x 添加标准差为 sigma 的高斯噪声。
    """
    noise = torch.randn_like(x) * sigma
    return x + noise

def visualize_results(original, noisy, denoised):
    """
    可视化原始数据、带噪数据与去噪结果图。
    """
    original_np = original.squeeze().cpu().numpy()
    noisy_np = noisy.squeeze().cpu().numpy()
    denoised_np = denoised.squeeze().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axs[0].imshow(original_np, cmap='viridis')
    axs[0].set_title('原始数据')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(noisy_np, cmap='viridis')
    axs[1].set_title('带噪数据')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    im2 = axs[2].imshow(denoised_np, cmap='viridis')
    axs[2].set_title('去噪结果')
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    plt.show()

# -------------------------
# 训练函数
# -------------------------
def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    mse_loss = nn.MSELoss()
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in dataloader:
            x = x.to(device)  # [batch_size, 1, 68, 150]
            batch_size = x.size(0)
            # 随机生成噪声水平 t（范围 [0,1]）
            t = torch.rand(batch_size, device=device)
            sigma = 0.1 + t * 0.9  # sigma 取值范围为 [0.1, 1.0]
            sigma_expanded = sigma.view(batch_size, 1, 1, 1)
            x_noisy = add_noise(x, sigma_expanded)
            optimizer.zero_grad()
            score_est = model(x_noisy, t)
            # 构造损失目标，梯度目标计算参考扩散模型原理
            target = - (x_noisy - x) / (sigma_expanded ** 2)
            loss = mse_loss(score_est, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")
    return epoch_losses

# -------------------------
# 主函数：训练、保存模型与测试去噪效果
# -------------------------
def main():
    data_dir = "low"  # 请确保此目录下有 .pt 数据文件
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PrecipitationDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample = dataset[0]
    in_channels = sample.shape[0]
    print(f"检测到的输入通道数: {in_channels}")

    model = ScoreNetOptimized(in_channels=in_channels, time_emb_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("开始训练……")
    losses = train(model, dataloader, optimizer, device, epochs=num_epochs)
    torch.save(model.state_dict(), "score_model_innovative_50.pth")
    print("模型已保存为 score_model_innovative_50.pth")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, num_epochs + 1), losses, marker='o')
    plt.title("训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # 测试去噪效果
    model.eval()
    with torch.no_grad():
        test_sample = dataset[0].unsqueeze(0).to(device)
        t_fixed = torch.tensor([0.5], device=device)
        sigma = 0.1 + t_fixed * 0.9
        sigma_expanded = sigma.view(1, 1, 1, 1)
        noisy_sample = add_noise(test_sample, sigma_expanded)
        score_est = model(noisy_sample, t_fixed)
        # 根据模型输出构造去噪结果
        denoised_sample = noisy_sample + (sigma_expanded ** 2) * score_est
    visualize_results(test_sample[0], noisy_sample[0], denoised_sample[0])

if __name__ == "__main__":
    main()