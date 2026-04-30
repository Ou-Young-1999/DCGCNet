import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
from collections import Counter

class AFDataset(Dataset):
    def __init__(self, path, orig_sr=500, target_sr=500,start=0,total=3):
        """
        Args:
            path (str): 数据目录路径，包含多个 .npz 文件
            orig_sr (int): 原始采样率 (Hz)
            target_sr (int): 目标采样率 (Hz)，若相同则跳过重采样
        """
        self.path = path
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.start = start
        self.total = total
        # 加载并预处理所有数据段
        if 'SPHDB' in path:
            self.X, self.y = self._load_and_preprocess_sphdb()
            # 转换标签为数字
            self.y = np.array([1 if label == 1.0 else 0 for label in self.y])
        else:
            self.X, self.y = self._load_and_preprocess()
            # 转换标签为数字
            self.y = np.array([1 if label == 'AFIB' else 0 for label in self.y])
        

    def _load_and_preprocess(self):
        all_X = []
        all_y = []
        nums = len(os.listdir(self.path))
        for fname in sorted(os.listdir(self.path))[nums//self.total*self.start:nums//self.total*(self.start+1)]:
            if not fname.endswith('.npz'):
                continue
            npz_path = os.path.join(self.path, fname)  # 修复：原代码漏了 fname！
            data = np.load(npz_path)
            X = data['segments']  # shape: [N, L, 2]
            y = data['labels']    # shape: [N,]

            if X.ndim != 3 or X.shape[-1] != 2:
                print(fname)
                continue

            # 遍历每一段进行预处理
            processed_segments = []
            for seg in X:
                processed_segments.append(seg)

            if processed_segments:
                all_X.append(np.stack(processed_segments, axis=0))  # [N, L, 2]
                all_y.append(y)

        if not all_X:
            raise ValueError("No valid data loaded from the path.")

        X_full = np.concatenate(all_X, axis=0)  # [Total_N, L, 2]
        y_full = np.concatenate(all_y, axis=0)  # [Total_N,]

        return X_full, y_full

    def _load_and_preprocess_sphdb(self):
        all_X = []
        all_y = []

        # 获取所有 data_g*.npy 和 label_g*.npy 文件
        data_files = sorted([f for f in os.listdir(self.path) if f.startswith('data_g') and f.endswith('.npy')])
        label_files = sorted([f for f in os.listdir(self.path) if f.startswith('label_g') and f.endswith('.npy')])

        # 确保数量一致
        assert len(data_files) == len(label_files), "Mismatch between data and label files!"
        assert len(data_files) > 0, "No data_g*.npy or label_g*.npy files found!"

        # 按组索引分片：比如 total=5, start=0 → 取 [0:1], 即第0组；start=2 → 取 [2:3]
        total_groups = len(data_files)
        group_indices = list(range(total_groups))
        
        # 计算当前进程/任务应处理的组范围
        start_idx = (total_groups * self.start) // self.total
        end_idx = (total_groups * (self.start + 1)) // self.total
        selected_group_indices = group_indices[start_idx:end_idx]

        for i in selected_group_indices:
            data_file = data_files[i]
            label_file = label_files[i]

            data_path = os.path.join(self.path, data_file)
            label_path = os.path.join(self.path, label_file)

            X = np.load(data_path)      # shape: [N, L]
            y = np.load(label_path)     # shape: [N,]


            # 只取前 2000 个采样点
            X = X[:, :2000]  # 截断到前 2000 点

            # 增加通道维度: [N, 2000] → [N, 2000, 1]
            X = np.expand_dims(X, axis=-1)

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            raise ValueError("No valid data loaded from the path.")

        X_full = np.concatenate(all_X, axis=0)  # [Total_N, 2000, 1]
        y_full = np.concatenate(all_y, axis=0)  # [Total_N,]

        return X_full, y_full

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]      # shape: [L, 2]
        y = self.y[idx]

        num_channels = x.shape[1]
        L_orig = x.shape[0]

        # ===== Step 1: 重采样（对每个通道独立处理）=====
        if self.orig_sr != self.target_sr:
            num_samples = int(L_orig * self.target_sr / self.orig_sr)
            x_resampled = np.zeros((num_samples, num_channels), dtype=np.float32)
            for ch in range(num_channels):
                x_resampled[:, ch] = resample(x[:, ch], num_samples)
            seg = x_resampled
        else:
            seg = x.astype(np.float32)

        # ===== Step 2: 每导联标准化（per-lead z-score）=====
        seg_normalized = np.zeros_like(seg)
        for ch in range(num_channels):
            mean = np.mean(seg[:, ch])
            std = np.std(seg[:, ch])
            if std > 1e-6:
                seg_normalized[:, ch] = (seg[:, ch] - mean) / std
            else:
                seg_normalized[:, ch] = seg[:, ch] - mean  # avoid division by zero

        # ===== Step 3: 扩展为 12 导联 [12, L] =====
        L_new = seg_normalized.shape[0]
        seg_12lead = np.zeros((12, L_new), dtype=np.float32)
        
        # 填充实际存在的导联：Lead I (idx=0), Lead II (idx=1)
        for ch in range(num_channels):
            seg_12lead[ch, :] = seg_normalized[:, ch]

        # 其余导联保持为 0（已初始化）

        seg_12lead = torch.tensor(seg_12lead, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return seg_12lead, y

