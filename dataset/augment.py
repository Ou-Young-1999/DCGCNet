import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class ECGAugmenter:
    def __init__(
            self,
            p_mask_any=0.5,  # 概率：是否进行导联 mask 增强
            max_leads_to_mask=11,  # 增强1：最多 mask 多少个导联（0～11）

            p_mask_interval=0.5,  # 概率：是否进行导联局部 mask 增强
            interval_len_range=(0.1,0.5),

            p_gaussian_noise=0.5,  # 概率：是否加高斯噪声
            noise_std_range=(0.05, 0.15),  # 噪声标准差范围（相对于信号幅值）

            p_baseline_wander=0.5,  # 概率：是否加基线漂移
            bw_amplitude_range=(0.1, 0.8),  # 漂移幅度范围（相对于信号幅值）
            bw_freq_range=(0.01, 0.5),  # 漂移频率范围（Hz）

            fs=250,  # 采样率（用于基线漂移生成）
            inplace=False  # 是否原地修改输入 tensor
    ):
        self.p_mask_any = p_mask_any
        self.max_leads_to_mask = max_leads_to_mask

        self.p_mask_interval = p_mask_interval
        self.interval_len_range = interval_len_range

        self.p_gaussian_noise = p_gaussian_noise
        self.noise_std_range = noise_std_range

        self.p_baseline_wander = p_baseline_wander
        self.bw_amplitude_range = bw_amplitude_range
        self.bw_freq_range = bw_freq_range

        self.fs = fs
        self.inplace = inplace

    def __call__(self, x):
        """
        Apply augmentations to a 12-lead ECG tensor of shape [12, L].
        Args:
            x (torch.Tensor): Input ECG with shape [12, L]
        Returns:
            torch.Tensor: Augmented ECG with same shape
        """
        if not self.inplace:
            x = x.clone()

        # 2. Gaussian noise
        if torch.rand(1).item() < self.p_gaussian_noise:
            x = self._add_gaussian_noise(x)

        # 3. Baseline wander
        if torch.rand(1).item() < self.p_baseline_wander:
            x = self._add_baseline_wander(x)

        # 1. Random lead masking
        if torch.rand(1).item() < self.p_mask_any:
            x = self._mask_leads(x)

        # 2. Random lead masking
        if torch.rand(1).item() < self.p_mask_interval:
            x = self._mask_interval(x)

        return x

    def _mask_leads(self, x):
        """Randomly mask 0 to max_leads_to_mask leads."""
        num_leads = x.shape[0]  # should be 12
        # Mask 1 to max_leads_to_mask leads
        n_mask = torch.randint(10, min(self.max_leads_to_mask, num_leads) + 1, (1,)).item()
        lead_indices = torch.randperm(num_leads)[:n_mask]
        x[lead_indices] = 0.0
        
        return x

    def _mask_interval(self, x):
        """Randomly mask a continuous interval in the signal."""
        num_leads, length = x.shape  # [12, L]
        min_interval_len_ratio,max_interval_len_ratio = self.interval_len_range

        # Calculate the actual interval lengths based on the ratios
        min_interval_len = int(min_interval_len_ratio * length)
        max_interval_len = int(max_interval_len_ratio * length)

        # # Randomly choose an interval length within the specified range
        # interval_len = torch.randint(min_interval_len, max_interval_len + 1, (1,)).item()
        #
        # # Randomly choose a starting point for the interval
        # start_point = torch.randint(0, length - interval_len + 1, (1,)).item()
        # end_point = start_point + interval_len
        #
        # # Mask the chosen interval by setting it to zero or another value
        # # Here we set it to zero, but you can also use mean or median values
        # x[:, start_point:end_point] = 0.0

        for i in range(12):
            # Randomly choose an interval length within the specified range
            interval_len = torch.randint(min_interval_len, max_interval_len + 1, (1,)).item()

            # Randomly choose a starting point for the interval
            start_point = torch.randint(0, length - interval_len + 1, (1,)).item()
            end_point = start_point + interval_len

            # Mask the chosen interval by setting it to zero or another value
            # Here we set it to zero, but you can also use mean or median values
            x[i, start_point:end_point] = 0.0

        return x

    def _add_gaussian_noise(self, x):
        """Add Gaussian noise with std sampled from range."""
        std_min, std_max = self.noise_std_range
        # Estimate signal scale per lead (robust to outliers)
        signal_scale = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-6  # [12, 1]
        noise_std = torch.empty_like(signal_scale).uniform_(std_min, std_max)
        noise = torch.randn_like(x) * (noise_std * signal_scale)
        x.add_(noise)
        return x

    def _add_baseline_wander(self, x):
        """Add low-frequency sinusoidal baseline wander."""
        L = x.shape[1]
        t = torch.linspace(0, L / self.fs, L, device=x.device)

        amp_min, amp_max = self.bw_amplitude_range
        freq_min, freq_max = self.bw_freq_range

        # Sample amplitude and frequency
        amplitude = torch.empty(x.shape[0], 1, device=x.device).uniform_(amp_min, amp_max)
        freq = torch.empty(x.shape[0], 1, device=x.device).uniform_(freq_min, freq_max)

        # Generate phase randomly
        phase = torch.empty(x.shape[0], 1, device=x.device).uniform_(0, 2 * np.pi)

        # Compute baseline wander: A * sin(2πft + φ)
        wander = amplitude * torch.sin(2 * np.pi * freq * t + phase)  # [12, L]

        # Scale wander relative to signal magnitude
        signal_scale = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-6
        wander = wander * signal_scale

        x.add_(wander)
        return x

# ==============================
# 独立增强函数（用于单独调用）
# ==============================

def mask_leads_only(x, max_leads_to_mask=11, fs=250):
    """仅做导联遮掩"""
    augmenter = ECGAugmenter(
        p_mask_any=1.0,
        max_leads_to_mask=max_leads_to_mask,
        p_mask_interval=0.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=0.0,
        fs=fs,
        inplace=False
    )
    return augmenter(x)

def mask_interval_only(x, interval_len_range=(0.1,0.5), fs=250):
    """仅做导联遮掩"""
    augmenter = ECGAugmenter(
        p_mask_any=0.0,
        p_mask_interval=1.0,
        max_leads_to_mask=interval_len_range,
        p_gaussian_noise=0.0,
        p_baseline_wander=0.0,
        fs=fs,
        inplace=False
    )
    return augmenter(x)

def add_gaussian_noise_only(x, noise_std_range=(0.05, 0.15), fs=250):
    """仅加高斯噪声"""
    augmenter = ECGAugmenter(
        p_mask_any=0.0,
        p_mask_interval=0.0,
        p_gaussian_noise=1.0,
        noise_std_range=noise_std_range,
        p_baseline_wander=0.0,
        fs=fs,
        inplace=False
    )
    return augmenter(x)

def add_baseline_wander_only(x, bw_amplitude_range=(0.1, 0.5), bw_freq_range=(0.05, 0.3), fs=250):
    """仅加基线漂移"""
    augmenter = ECGAugmenter(
        p_mask_any=0.0,
        p_mask_interval=0.0,
        p_gaussian_noise=0.0,
        p_baseline_wander=1.0,
        bw_amplitude_range=bw_amplitude_range,
        bw_freq_range=bw_freq_range,
        fs=fs,
        inplace=False
    )
    return augmenter(x)

def apply_all_augmentations(x, **kwargs):
    """应用全部增强"""
    augmenter = ECGAugmenter(
        p_mask_any=1.0,
        p_mask_interval=1.0,
        p_gaussian_noise=1.0,
        p_baseline_wander=1.0,
        **kwargs
    )
    return augmenter(x)


# ==============================
# Z-score 标准化（每导联）
# ==============================

def z_score_normalize(x):
    """
    对 ECG 每个导联做 Z-score 标准化（均值为0，标准差为1）
    Args:
        x (torch.Tensor): shape [12, L]
    Returns:
        torch.Tensor: normalized ECG, same shape
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


# ==============================
# 下采样到目标采样率（如 500 -> 100 Hz）
# ==============================

def downsample_ecg(x, orig_fs=500, target_fs=100):
    """
    使用简单整数下采样（要求 orig_fs % target_fs == 0）
    Args:
        x (torch.Tensor): shape [12, L]
        orig_fs (int): 原始采样率
        target_fs (int): 目标采样率
    Returns:
        torch.Tensor: downsampled ECG, shape [12, L_new]
    """
    if orig_fs % target_fs != 0:
        raise ValueError(f"Only integer downsampling supported. {orig_fs} not divisible by {target_fs}")

    factor = orig_fs // target_fs  # 500 -> 100 => factor = 5
    # 取每隔 factor 个点（简单下采样，实际可加抗混叠滤波）
    return x[:, ::factor]


# ==============================
# 可视化函数
# ==============================

def plot_ecg_comparison(original, augmented, title, save_path, fs=500, leads_to_show=12):
    """
    绘制原始 vs 增强后的 ECG（前 N 个导联）
    """
    L = original.shape[1]
    time_sec = torch.linspace(0, L / fs, L).numpy()

    fig, axes = plt.subplots(leads_to_show, 1, figsize=(12, 2 * leads_to_show), sharex=True)
    if leads_to_show == 1:
        axes = [axes]

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for i in range(leads_to_show):
        axes[i].plot(time_sec, original[i].numpy(), color='black', linewidth=1.2, label='Original')
        axes[i].plot(time_sec, augmented[i].numpy(), color='red', linewidth=0.8, alpha=0.5, label='Augmented')
        axes[i].set_ylabel(lead_names[i], fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==============================
# 主程序：测试与可视化
# ==============================

if __name__ == "__main__":

    pt_path = "F:\\CPSC2018\\PTDB\\seg_000015.pt"
    output_dir = "./augmentation_results"
    fs = 500

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载 ECG
    ecg_tensor = torch.load(pt_path)  # shape [12, L]
    if ecg_tensor.ndim != 2 or ecg_tensor.shape[0] != 12:
        raise ValueError(f"Expected [12, L] tensor, got {ecg_tensor.shape}")

    print(f"Loaded ECG with shape: {ecg_tensor.shape}, duration: {ecg_tensor.shape[1] / fs:.1f} seconds")


    # 1. 仅遮掩
    masked = mask_leads_only(ecg_tensor.clone(), fs=fs)
    plot_ecg_comparison(
        ecg_tensor, masked,
        title="ECG Augmentation: Lead Masking Only",
        save_path=os.path.join(output_dir, "masked.png"),
        fs=fs
    )
    masked = mask_interval_only(ecg_tensor.clone(), fs=fs)
    plot_ecg_comparison(
        ecg_tensor, masked,
        title="ECG Augmentation: Lead Interval Masking Only",
        save_path=os.path.join(output_dir, "maskedinterval.png"),
        fs=fs
    )

    # 2. 仅高斯噪声
    noised = add_gaussian_noise_only(ecg_tensor.clone(), fs=fs)
    plot_ecg_comparison(
        ecg_tensor, noised,
        title="ECG Augmentation: Gaussian Noise Only",
        save_path=os.path.join(output_dir, "noised.png"),
        fs=fs
    )

    # 3. 仅基线漂移
    wandered = add_baseline_wander_only(ecg_tensor.clone(), fs=fs)
    plot_ecg_comparison(
        ecg_tensor, wandered,
        title="ECG Augmentation: Baseline Wander Only",
        save_path=os.path.join(output_dir, "wandered.png"),
        fs=fs
    )

    # 4. 全部增强
    all_aug = apply_all_augmentations(ecg_tensor.clone(), fs=fs)
    plot_ecg_comparison(
        ecg_tensor, all_aug,
        title="ECG Augmentation: All (Mask + Noise + Wander)",
        save_path=os.path.join(output_dir, "all_aug.png"),
        fs=fs
    )

    # 5. Z-score 标准化
    zscored = z_score_normalize(ecg_tensor.clone())
    plot_ecg_comparison(
        zscored, zscored,
        title="ECG Preprocessing: Z-Score Normalization (per lead)",
        save_path=os.path.join(output_dir, "zscore.png"),
        fs=fs
    )

    # 6. 下采样到 100 Hz
    downsampled = downsample_ecg(ecg_tensor.clone(), orig_fs=fs, target_fs=100)
    print(f"Downsampled from {ecg_tensor.shape[1]} to {downsampled.shape[1]} samples (500Hz → 100Hz)")

    # 可视化下采样（需调整时间轴）
    L_orig = ecg_tensor.shape[1]
    L_new = downsampled.shape[1]
    time_orig = torch.linspace(0, L_orig / fs, L_orig).numpy()
    time_new = torch.linspace(0, L_new / 100, L_new).numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)  # 只画前3导联避免太密
    lead_names = ['I', 'II', 'V1']
    for i in range(3):
        axes[i].plot(time_orig, ecg_tensor[i].numpy(), color='black', linewidth=1.0, label='Original (500Hz)')
        axes[i].scatter(time_new, downsampled[i].numpy(), color='red', s=8, alpha=0.7, label='Downsampled (100Hz)')
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle("ECG Downsampling: 500 Hz → 100 Hz")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "downsampled.png"), dpi=150)
    plt.close()

    print(f"✅ All augmentation visualizations saved to: {output_dir}")