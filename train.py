# train_multitask.py

import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
import numpy as np
import random
from tqdm import tqdm
from dataset.dataloader import ECGDataset
from dataset.augment import ECGAugmenter
from model.unify import ECG_VQ_Graph, calculate_loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_recon_metrics(preds, targets):
    """Compute PCC, MAE, RMSE for reconstruction"""
    preds = preds.detach().cpu().numpy()      # (B, C, T)
    targets = targets.detach().cpu().numpy()  # (B, C, T)

    diff = preds - targets
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    B, C, T = preds.shape
    pccs = []
    for b in range(B):
        for c in range(C):
            p, _ = pearsonr(preds[b, c], targets[b, c])
            if not np.isnan(p):
                pccs.append(p)
    pcc = np.mean(pccs) if pccs else 0.0
    return pcc, mae, rmse


def train_one_epoch(model, dataloader, optimizer, device, lambda_dict):
    model.train()
    # if hasattr(model, 'vq_morph'):
    #     model.vq_morph.use_vqbridge = True
    # if hasattr(model, 'vq_rhythm'):
    #     model.vq_rhythm.use_vqbridge = True
    total_loss = 0.0
    total_cls_loss = 0.0
    total_rec_loss = 0.0
    total_vq_loss = 0.0
    total_con_local = 0.0
    total_con_global = 0.0

    all_preds = []
    all_labels = []
    total_pcc = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0

    for ecg_input, ecg_aug, ecg, labels, _ in tqdm(dataloader, desc="Training"):
        ecg_input = ecg_input.to(device)      # [B, 12, 2500] — 已 mask
        ecg_aug = ecg_aug.to(device)
        labels = labels.to(device)            # [B,] — 房颤标签
        ecg_target = ecg.to(device)        # 重构目标是原始输入

        optimizer.zero_grad()

        outputs = model(ecg_input,ecg_aug)
        loss, loss_details = calculate_loss(
            outputs,
            targets_af=labels,
            x_original=ecg_target,
            lambda_cls=lambda_dict['cls'],
            lambda_rec=lambda_dict['rec'],
            lambda_vq=lambda_dict['vq'],
            lambda_global_con=lambda_dict['global_con'],
            lambda_local_con=lambda_dict['local_con'],
        )

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_cls_loss += loss_details['cls']
        total_rec_loss += loss_details['rec']
        total_vq_loss += loss_details['vq']
        total_con_local += loss_details['con_local']
        total_con_global += loss_details['con_global']

        # Classification metrics
        preds = torch.argmax(outputs['logits_af'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Reconstruction metrics
        pcc, mae, rmse = compute_recon_metrics(outputs['x_recon'], ecg_target)
        total_pcc += pcc
        total_mae += mae
        total_rmse += rmse

        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_pcc = total_pcc / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    metrics = {
        'loss': avg_loss,
        'cls_loss': total_cls_loss / num_batches,
        'rec_loss': total_rec_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'con_local': total_con_local / num_batches,
        'con_global': total_con_global / num_batches,
        'acc': acc,
        'f1': f1,
        'pcc': avg_pcc,
        'mae': avg_mae,
        'rmse': avg_rmse
    }
    return metrics


@torch.no_grad()
def validate(model, dataloader, device, lambda_dict):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_rec_loss = 0.0
    total_vq_loss = 0.0
    total_con_local = 0.0
    total_con_global = 0.0

    all_preds = []
    all_labels = []
    total_pcc = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0

    for ecg_input, ecg_aug, ecg, labels, _ in tqdm(dataloader, desc="Validation"):
        ecg_input = ecg_input.to(device)
        ecg_aug = ecg_aug.to(device)
        labels = labels.to(device)
        ecg_target = ecg.to(device)

        outputs = model(ecg_input,ecg_aug)
        loss, loss_details = calculate_loss(
            outputs,
            targets_af=labels,
            x_original=ecg_target,
            lambda_cls=lambda_dict['cls'],
            lambda_rec=lambda_dict['rec'],
            lambda_vq=lambda_dict['vq'],
            lambda_global_con=lambda_dict['global_con'],
            lambda_local_con=lambda_dict['local_con'],
        )

        total_loss += loss.item()
        total_cls_loss += loss_details['cls']
        total_rec_loss += loss_details['rec']
        total_vq_loss += loss_details['vq']
        total_con_local += loss_details['con_local']
        total_con_global += loss_details['con_global']

        preds = torch.argmax(outputs['logits_af'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pcc, mae, rmse = compute_recon_metrics(outputs['x_recon'], ecg_target)
        total_pcc += pcc
        total_mae += mae
        total_rmse += rmse
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_pcc = total_pcc / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    metrics = {
        'loss': avg_loss,
        'cls_loss': total_cls_loss / num_batches,
        'rec_loss': total_rec_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'con_local': total_con_local / num_batches,
        'con_global': total_con_global / num_batches,
        'acc': acc,
        'f1': f1,
        'pcc': avg_pcc,
        'mae': avg_mae,
        'rmse': avg_rmse
    }
    return metrics


def main():
    # Load config
    with open("config/multitask_chapman_1.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Augmentation (masking simulates missing leads)
    train_aug = ECGAugmenter(
        p_mask_any=cfg["augmentation"]["p_mask_any"],
        max_leads_to_mask=cfg["augmentation"]["max_leads_to_mask"],
        p_mask_interval=cfg["augmentation"]["p_mask_interval"],
        interval_len_range=cfg["augmentation"]["interval_len_range"],
        p_gaussian_noise=cfg["augmentation"]["p_gaussian_noise"],
        noise_std_range=cfg["augmentation"]["noise_std_range"],
        p_baseline_wander=cfg["augmentation"]["p_baseline_wander"],
        bw_amplitude_range=cfg["augmentation"]["bw_amplitude_range"],
        bw_freq_range=cfg["augmentation"]["bw_freq_range"],
        fs=cfg["augmentation"]["fs"],
        inplace=True
    )

    # Datasets
    train_dataset = ECGDataset(
        csv_path=cfg["data"]["train_csv"],
        data_root_dir=cfg["data"]["data_root_dir"],
        transform=None,
        oversample=cfg["training"]["oversample"],
        random_seed=cfg["training"]["seed"]
    )
    val_dataset = ECGDataset(
        csv_path=cfg["data"]["val_csv"],
        data_root_dir=cfg["data"]["data_root_dir"],
        transform=None,
        oversample=False
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)

    # Model
    model = ECG_VQ_Graph(
        input_channels=cfg["model"].get("input_channels", 12),
        seq_len=cfg["model"].get("seq_len", 2500),
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        codebook_size=cfg["model"].get("codebook_size", 512)
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # Loss weights
    lambda_dict = {
        'cls': 1.0,
        'rec': 1.0,
        'vq': 0.2,
        'local_con': 0.1,      # 局部对比损失权重
        'global_con': 0.5,     # 全局对比损失权重
    }

    # Training loop
    best_val_metric = -float('inf')  # e.g., use F1 or PCC
    metric_to_monitor = cfg["training"].get("monitor_metric", "pcc")  # 'f1', 'pcc', 'acc'

    for epoch in range(cfg["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['num_epochs']}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, lambda_dict)
        val_metrics = validate(model, val_loader, device, lambda_dict)

        # Print metrics
        print(f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f} | PCC: {train_metrics['pcc']:.4f}")
        print(f"      | Con Local: {train_metrics['con_local']:.4f} | Con Global Loss: {train_metrics['con_global']:.4f} | Rec Loss: {train_metrics['rec_loss']:.4f}")
        print(f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | PCC: {val_metrics['pcc']:.4f}")
        print(f"      | Cls Loss: {val_metrics['cls_loss']:.4f} | Rec Loss: {val_metrics['rec_loss']:.4f} | VQ Loss: {val_metrics['vq_loss']:.4f}")

        # Save best model
        current_metric = val_metrics[metric_to_monitor]
        if cfg["training"]["save_checkpoint"] and current_metric > best_val_metric:
            best_val_metric = current_metric
            os.makedirs(os.path.dirname(cfg["training"]["checkpoint_path"]), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'best_{metric_to_monitor}': best_val_metric
            }, cfg["training"]["checkpoint_path"])
            print(f"✅ Best model saved ({metric_to_monitor}: {best_val_metric:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    main()