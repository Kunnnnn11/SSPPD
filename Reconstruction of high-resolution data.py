import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

# 设置全局字体（Microsoft YaHei）并确保负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------------
# 改进后的正弦时间嵌入模块（注册缓冲区以存储频率向量）
# ---------------------------
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
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        sinusoid_inp = t * self.inv_freq.unsqueeze(0)
        sin_emb = torch.sin(sinusoid_inp)
        cos_emb = torch.cos(sinusoid_inp)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        return emb

# ---------------------------
# 辅助函数：中心裁剪（用于 U-Net 拼接时对齐尺寸）
# ---------------------------
def center_crop(tensor, target_h, target_w):
    _, _, h, w = tensor.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return tensor[:, :, start_h: start_h + target_h, start_w: start_w + target_w]

# ---------------------------
# 模型定义：残差块、下采样、上采样与 U-Net 结构
# ---------------------------
class ResBlock(nn.Module):
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
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)

class ScoreNetOptimized(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super(ScoreNetOptimized, self).__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.resblock1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.down1 = DownSample(base_channels)
        self.resblock2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownSample(base_channels * 2)
        self.resblock3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.up1 = UpSample(base_channels * 4)
        self.resblock4 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = UpSample(base_channels * 2)
        self.resblock5 = ResBlock(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t.view(-1, 1))
        x1 = self.resblock1(x, t_emb)
        x2 = self.down1(x1)
        x2 = self.resblock2(x2, t_emb)
        x3 = self.down2(x2)
        x3 = self.resblock3(x3, t_emb)
        x_up = self.up1(x3)
        _, _, h2, w2 = x2.shape
        x_up = center_crop(x_up, h2, w2)
        x_cat = torch.cat([x_up, x2], dim=1)
        x_mid = self.resblock4(x_cat, t_emb)
        x_up2 = self.up2(x_mid)
        _, _, h1, w1 = x1.shape
        x_up2 = center_crop(x_up2, h1, w1)
        x_cat2 = torch.cat([x_up2, x1], dim=1)
        x_out = self.resblock5(x_cat2, t_emb)
        score = self.final_conv(x_out)
        return score

# ---------------------------
# 数据集定义：低分辨率降水数据（precipitation）文件夹
# ---------------------------
class PrecipitationDataset(torch.utils.data.Dataset):
    def __init__(self, precip_dir):
        self.file_paths = glob.glob(os.path.join(precip_dir, "*.pt"))
        # 排除含有 "uwind" 的文件，确保只包含降水数据
        self.file_paths = [fp for fp in self.file_paths if "uwind" not in os.path.basename(fp)]
        if not self.file_paths:
            raise ValueError("在指定的降水数据目录中未找到符合条件的 .pt 文件。")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sample = torch.load(file_path)
        if sample.dim() == 2:
            sample = sample.unsqueeze(0)
        sample[torch.isnan(sample)] = 0
        original_min = sample.min()
        original_max = sample.max()
        # 归一化：仅用于模型处理，后续根据原始值域映射回去
        sample_normalized = (sample - sample.mean()) / (sample.std() + 1e-8)
        return {
            "data": sample_normalized,
            "original_min": original_min,
            "original_max": original_max,
            "file_name": os.path.basename(file_path)
        }

# ---------------------------
# 辅助工具函数：添加噪声、质量守恒以及结果可视化
# ---------------------------
def add_noise(x, sigma):
    noise = torch.randn_like(x) * sigma
    return x + noise

def mass_conservation_penalty(high_res, low_res, low_res_shape):
    high_res_downsampled = F.interpolate(high_res,
                                         size=(low_res_shape[2], low_res_shape[3]),
                                         mode="bilinear",
                                         align_corners=False)
    penalty = (low_res.sum() - high_res_downsampled.sum()).abs()
    return penalty

def guided_sampling(model, terrain, wind, low_res,
                    low_res_shape, high_res_shape,
                    num_steps=500, step_size=0.0005,
                    guidance_weight_terrain=0.1,
                    guidance_weight_u=0.1, guidance_weight_v=0.1, guidance_weight_amp=0.1,
                    consistency_weight=0.1, mass_conservation_weight=0.01,
                    device="cpu"):
    """
    采样生成高分辨率数据：
      - model: 预训练模型
      - terrain: 归一化后的地形数据，与 low_res 尺寸对应
      - wind: 一个元组 (u, v, amp)，分别为归一化后的风速 u 分量、v 分量和风速幅值，均与 low_res 尺寸对应
      - low_res: 低分辨率降水数据
      - low_res_shape / high_res_shape: 数据尺寸，格式为 (1,1,H,W)
      - 各权重参数控制辅助项的影响
    """
    # 从低分辨率降水上采样后添加噪声，作为初始候选
    x = F.interpolate(low_res, size=(high_res_shape[2], high_res_shape[3]),
                      mode="bilinear", align_corners=False)
    x = add_noise(x, sigma=0.01)
    # 解包风速数据的 u, v 和幅值
    u, v, amp = wind
    with torch.no_grad():
        t_schedule = torch.linspace(1.0, 0.01, num_steps, device=device)
        for i, t in enumerate(t_schedule):
            batch_size = x.size(0)
            x_down = F.interpolate(x, size=(low_res_shape[2], low_res_shape[3]),
                                   mode="bilinear", align_corners=False)
            t_tensor = t.repeat(batch_size)
            score_low = model(x_down, t_tensor)
            score_up = F.interpolate(score_low, size=(high_res_shape[2], high_res_shape[3]),
                                     mode="bilinear", align_corners=False)
            # 调整地形数据到高分辨率尺寸
            terrain_resized = F.interpolate(terrain.unsqueeze(0),
                                            size=(high_res_shape[2], high_res_shape[3]),
                                            mode="bilinear", align_corners=False)
            guidance_terrain = guidance_weight_terrain * (terrain_resized - x)
            # 调整 u, v, amp 至高分辨率尺寸
            u_resized = F.interpolate(u, size=(high_res_shape[2], high_res_shape[3]),
                                      mode="bilinear", align_corners=False)
            v_resized = F.interpolate(v, size=(high_res_shape[2], high_res_shape[3]),
                                      mode="bilinear", align_corners=False)
            amp_resized = F.interpolate(amp, size=(high_res_shape[2], high_res_shape[3]),
                                        mode="bilinear", align_corners=False)
            guidance_wind = (guidance_weight_u * (u_resized - x) +
                             guidance_weight_v * (v_resized - x) +
                             guidance_weight_amp * (amp_resized - x))
            # 保持低分辨率数据与当前候选结果之间一致性
            consistency_error = low_res - x_down
            consistency_guidance = consistency_weight * F.interpolate(consistency_error,
                                                                      size=(high_res_shape[2], high_res_shape[3]),
                                                                      mode="bilinear", align_corners=False)
            mass_penalty = mass_conservation_penalty(x, low_res, low_res_shape)
            mass_guidance = mass_conservation_weight * mass_penalty
            x = x + step_size * (score_up + guidance_terrain + guidance_wind + consistency_guidance - mass_guidance)
            noise_std = step_size ** 0.5
            x = x + noise_std * torch.randn_like(x)
            if (i + 1) % 100 == 0:
                print(f"采样步 {i + 1}/{num_steps}, t={t.item():.4f}, 质量守恒误差：{mass_penalty.item():.4f}")
                torch.cuda.empty_cache()
    return x

def visualize_sample(sample, title="Sample", cmap="viridis"):
    sample_np = sample.detach().squeeze().cpu().numpy()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(sample_np, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

def map_to_range(high_res_data, orig_min, orig_max):
    # 根据原始最小值和最大值将生成数据映射回原始值域
    high_min = high_res_data.min()
    high_max = high_res_data.max()
    high_res_mapped = (high_res_data - high_min) / (high_max - high_min + 1e-8) * (orig_max - orig_min) + orig_min
    return high_res_mapped

def load_wind_data(wind_file_path):
    """
    加载风速数据，并同时考虑 u 分量、v 分量以及幅值。
    预期风速数据尺寸为 [2, H, W]，其中第 0 个通道为 u，第 1 个通道为 v。
    返回一个元组 (u_norm, v_norm, amp_norm)，其中各张量形状均为 [1,1,H,W]，同时将 nan 值替换为 0。
    """
    wind_data = torch.load(wind_file_path)
    if wind_data.dim() == 3 and wind_data.size(0) == 2:
        u = wind_data[0]
        v = wind_data[1]
        amp = torch.sqrt(u**2 + v**2)
        # 调整尺寸为 [1,1,H,W]
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        amp = amp.unsqueeze(0).unsqueeze(0)
        # 替换 nan 值为 0
        u[torch.isnan(u)] = 0
        v[torch.isnan(v)] = 0
        amp[torch.isnan(amp)] = 0
        # 分别归一化处理
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()
        amp_min, amp_max = amp.min(), amp.max()
        u_norm = (u - u_min) / (u_max - u_min + 1e-8)
        v_norm = (v - v_min) / (v_max - v_min + 1e-8)
        amp_norm = (amp - amp_min) / (amp_max - amp_min + 1e-8)
        return (u_norm, v_norm, amp_norm)
    else:
        raise ValueError("风速数据格式错误，预期尺寸为 [2, H, W]。")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 分别指定降水数据和风速数据所在文件夹
    precip_dir = "low"  # 降水数据文件夹（文件名格式：clipped_ERA5_global_daily_20230101.pt）
    wind_dir = "wind"      # 风速数据文件夹（文件名格式：clipped_ERA5_global_uwind_daily_20230101.pt）
    output_dir = "generated_all_dem_wind_680_1500"  # 保存生成的高分辨率数据
    os.makedirs(output_dir, exist_ok=True)

    dataset = PrecipitationDataset(precip_dir)
    num_samples = len(dataset)
    print(f"降水样本总数：{num_samples}")

    # ---------------------------
    # 加载地形数据并归一化处理
    # ---------------------------
    terrain = torch.load("combined_dem.pt", map_location="cpu")
    terrain[torch.isnan(terrain)] = 0
    terrain = terrain.float()
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    terrain = (terrain - terrain_min) / (terrain_max - terrain_min + 1e-8)
    terrain = terrain.to(device)

    # 数据尺寸设置（例如低分辨率 [1,1,68,150]，高分辨率 [1,1,136,300]）
    low_res_shape = (1, 1, 68, 150)
    #high_res_shape = (1, 1, 136, 300)
    high_res_shape = (1, 1, 680, 1500)

    # 加载预训练模型并设置 eval 模式
    model = ScoreNetOptimized(in_channels=1).to(device)
    model.load_state_dict(torch.load("score_model_innovative.pth", map_location=device))
    model.eval()
    print("预训练打分模型加载完成。")

    for idx in range(num_samples):
        sample_dict = dataset[idx]
        file_name = sample_dict["file_name"]
        print(f"正在处理降水样本 {file_name}")
        low_res_sample = sample_dict["data"].unsqueeze(0).to(device)  # [1,1,68,150]
        orig_min = sample_dict["original_min"]
        orig_max = sample_dict["original_max"]

        # 根据降水数据文件名构造对应风速数据文件名（只依据日期匹配）
        # 示例：将 "clipped_ERA5_global_daily_20230101.pt" 转换为 "clipped_ERA5_global_uwind_daily_20230101.pt"
        wind_file_name = file_name.replace("global_daily", "global_uwind_daily")
        wind_file_path = os.path.join(wind_dir, wind_file_name)
        if not os.path.exists(wind_file_path):
            print(f"风速数据文件 {wind_file_name} 不存在，跳过该样本。")
            continue
        wind = load_wind_data(wind_file_path)
        # 将 wind 中的各个张量移动到 device 上
        wind = tuple(t.to(device) for t in wind)

        # 生成高分辨率数据（结合地形与风速辅助信息，同时考虑 u、v 以及幅值）
        x_gen = guided_sampling(model, terrain, wind, low_res_sample,
                                low_res_shape, high_res_shape,
                                num_steps=500, step_size=0.0005,
                                guidance_weight_terrain=0.1,
                                guidance_weight_u=0.1, guidance_weight_v=0.1, guidance_weight_amp=0.1,
                                consistency_weight=0.1,
                                mass_conservation_weight=0.01,
                                device=device)
        # 将生成结果映射回原降水数据的值域
        x_gen_mapped = map_to_range(x_gen, orig_min, orig_max)
        output_path = os.path.join(output_dir, file_name)
        torch.save(x_gen_mapped.cpu(), output_path)
        if idx % 100 == 0:
            visualize_sample(x_gen_mapped, title=f"样本 {file_name}")

    print("所有高分辨率数据生成完成.")

if __name__ == '__main__':
    main()