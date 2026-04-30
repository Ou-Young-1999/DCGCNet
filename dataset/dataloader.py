import os
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataset.augment import ECGAugmenter, z_score_normalize, downsample_ecg
from dataset.oversample import balance_binary_labels

class ECGDataset(Dataset):
    def __init__(self, csv_path, data_root_dir, transform=None, oversample=False, random_seed=42):
        """
        Args:
            csv_path (str): 路径到包含 segment_id 和 label 的 CSV 文件
            data_root_dir (str): 存放 .pt 文件的根目录
            transform (callable, optional): 数据增强函数
            oversample (bool): 是否开启二分类过采样
            random_seed (int): 随机种子
        """
        self.csv_path = csv_path
        self.data_root_dir = data_root_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        segment_ids = df['segment_id'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()

        if oversample:
            print(f"[ECGDataset] Oversampling enabled. Balancing classes...")
            segment_ids, labels = balance_binary_labels(segment_ids, labels, random_seed)
        else:
            counts = Counter(labels)
            print(f"[ECGDataset] Oversampling disabled. Class counts: {dict(counts)}")

        self.segment_ids = segment_ids
        self.labels = labels

    def __len__(self):
        return len(self.segment_ids)

    def __getitem__(self, idx):
        idx = idx % len(self.segment_ids)
        segment_id = self.segment_ids[idx] 
        label = self.labels[idx]

        pt_path = os.path.join(self.data_root_dir, f"{segment_id}.pt")
        ecg = torch.load(pt_path)  # shape expected: (12, L)
        ecg = ecg.float()

        ecg = z_score_normalize(ecg) # 标准化
        ecg = downsample_ecg(ecg, orig_fs=500, target_fs=250) # 采样频率

        # 构造输入，导联缺失
        if 'train' in self.csv_path:
            ecg_input = torch.zeros_like(ecg)  # shape: [12, seq_len] 或类似

            rand_choice = torch.rand(1).item()
            min_leads = 1
            max_leads = 12
            num_leads_to_keep = torch.randint(min_leads, max_leads + 1, (1,)).item()
            randperm = torch.randperm(12)
            lead_indices = randperm[:num_leads_to_keep]
            last_indices = randperm[num_leads_to_keep:]
            ecg_input[lead_indices] = ecg[lead_indices]

            train_augmenter = ECGAugmenter(
                p_mask_any=0.0,  # 概率：是否进行导联 mask 增强
                max_leads_to_mask=11,  # 增强1：最多 mask 多少个导联（0～11）

                p_mask_interval=0.9,  # 概率：是否进行导联局部 mask 增强
                interval_len_range=(0.1, 0.8),

                p_gaussian_noise=0.9,  # 概率：是否加高斯噪声
                noise_std_range=(0.1, 0.8),  # 噪声标准差范围（相对于信号幅值）

                p_baseline_wander=0.9,  # 概率：是否加基线漂移
                bw_amplitude_range=(0.1, 0.8),  # 漂移幅度范围（相对于信号幅值）
                bw_freq_range=(0.05, 0.3),  # 漂移频率范围（Hz）

                fs=250,  # 采样率（用于基线漂移生成）
                inplace=False  # 是否原地修改输入 tensor
            )
            ecg_aug = torch.zeros_like(ecg)
            ecg_aug[lead_indices] = ecg[lead_indices]
            ecg_aug = train_augmenter(ecg_aug)
            ecg_aug[last_indices] = 0
            return ecg_input, ecg_aug, ecg, label, segment_id

        # 构造输入：仅保留第0导联（第一导联），其余置0
        ecg_input = torch.zeros_like(ecg)
        ecg_input = ecg
        # ecg_input[0] = ecg[0]  # 保留第一导联
        # ecg_input[1] = ecg[1]  # 保留第二导联
        # ecg_input[6] = ecg[6]  # 保留第七导联

        return ecg_input, ecg_input, ecg, label, segment_id