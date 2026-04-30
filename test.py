# test_multitask.py

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, r2_score
)
from scipy.stats import pearsonr
import pandas as pd

from dataset.dataloader import ECGDataset
from model.unify import ECG_VQ_Graph


@torch.no_grad()
def test_multitask(model, dataloader, device, n_visualize=5):
    model.eval()
    if hasattr(model, 'vq_morph'):
        model.vq_morph.use_vqbridge = False
    if hasattr(model, 'vq_rhythm'):
        model.vq_rhythm.use_vqbridge = False

    # Classification
    all_labels = []
    all_preds = []
    all_probs = []

    # Reconstruction
    all_recon = []
    all_targets = []
    all_ids = []
    all_perplexity = []

    # Visualization
    visualize_data = []

    for i, (ecg_input, ecg_aug, ecg_target, labels, seg_ids) in enumerate(tqdm(dataloader)):
        ecg_input = ecg_input.to(device)  # [B, 12, T] — masked input
        ecg_aug = ecg_aug.to(device)
        ecg_target = ecg_target.to(device)  # [B, 12, T] — full original signal (for recon target)
        labels = labels.to(device)  # [B,] — AF label

        outputs = model(ecg_input,ecg_aug)
        logits_af = outputs['logits_af']  # [B, 2]
        x_recon = outputs['x_recon']  # [B, 12, T]
        perplexity = outputs['perplexity']

        # === Classification ===
        probs = torch.softmax(logits_af, dim=1)[:, 1].cpu().numpy()  # prob of class 1 (AF)
        preds = torch.argmax(logits_af, dim=1).cpu().numpy()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

        # === Reconstruction ===
        all_recon.append(x_recon.cpu())
        all_targets.append(ecg_target.cpu())
        all_ids.extend(seg_ids)
        all_perplexity.append(perplexity.cpu().numpy())
        # all_perplexity.append(0)

        # === Save for visualization ===
        if len(visualize_data) < n_visualize:
            B = x_recon.size(0)
            need = n_visualize - len(visualize_data)
            take = min(need, B)
            for j in range(take):
                visualize_data.append({
                    'id': seg_ids[j],
                    'label': labels[j].item(),
                    'target': ecg_target[j].cpu().numpy(),
                    'recon': x_recon[j].cpu().numpy()
                })

    # Concatenate reconstruction tensors
    recon_np = torch.cat(all_recon, dim=0).numpy()  # (N, 12, T)
    target_np = torch.cat(all_targets, dim=0).numpy()  # (N, 12, T)

    return {
        'labels': np.array(all_labels),
        'preds': np.array(all_preds),
        'probs': np.array(all_probs),
        'recon': recon_np,
        'target': target_np,
        'ids': all_ids,
        'perplexity': all_perplexity,
        'visualize': visualize_data
    }


def compute_recon_metrics(preds, targets):
    N, C, T = preds.shape
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    mae = np.mean(np.abs(preds_flat - targets_flat))
    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))
    r2 = r2_score(targets_flat, preds_flat)

    pcc_list = []
    for i in range(N):
        for c in range(C):
            x, y = preds[i, c], targets[i, c]
            if np.ptp(x) < 1e-8 or np.ptp(y) < 1e-8:
                continue
            p, _ = pearsonr(x, y)
            if not np.isnan(p):
                pcc_list.append(p)
    pcc = np.mean(pcc_list) if pcc_list else 0.0

    return float(pcc), float(mae), float(rmse), float(r2)


def plot_ecg_comparison(target, recon, seg_id, label, save_path, fs=250):
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    T = target.shape[1]
    time = np.arange(T) / fs  # seconds

    fig, axes = plt.subplots(12, 1, figsize=(16, 12), sharex=True)
    for i in range(12):
        axes[i].plot(time, target[i], color='black', linewidth=0.9, label='Original')
        axes[i].plot(time, recon[i], color='red', linewidth=0.9, linestyle='--', label='Reconstructed')
        axes[i].set_ylabel(leads[i], fontsize=9)
        axes[i].grid(True, linestyle=':', alpha=0.5)
        if i == 0:
            axes[i].legend(loc='upper right')

    plt.suptitle(f'ECG Reconstruction | ID: {seg_id} | Label: {"AF" if label == 1 else "Non-AF"}', fontsize=14)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # Load config
    with open("config/multitask_cpsc_1.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test dataset
    test_dataset = ECGDataset(
        csv_path=cfg["data"]["test_csv"],
        data_root_dir=cfg["data"]["data_root_dir"],
        transform=None,
        oversample=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # Model
    model = ECG_VQ_Graph(
        input_channels=cfg["model"].get("input_channels", 12),
        seq_len=cfg["model"].get("seq_len", 2500),
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        codebook_size=cfg["model"].get("codebook_size", 512)
    ).to(device)

    # Load checkpoint
    ckpt_path = cfg["training"]["checkpoint_path"]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # support legacy single-state-dict
    print(f"Loaded model from {ckpt_path}")

    # Run test
    results = test_multitask(model, test_loader, device, n_visualize=5)

    # === Classification Metrics ===
    labels = results['labels']
    preds = results['preds']
    probs = results['probs']

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    auc = roc_auc_score(labels, probs)

    print("\n" + "=" * 50)
    print("📊 CLASSIFICATION RESULTS (Atrial Fibrillation Detection)")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    report_str = classification_report(labels, preds, target_names=["Non-AF", "AF"], digits=4)
    print(report_str)

    # === Reconstruction Metrics ===
    pcc, mae, rmse, r2 = compute_recon_metrics(results['recon'], results['target'])
    mean_perplexity = sum(results['perplexity']) / len(results['perplexity'])
    print("\n" + "=" * 50)
    print("📈 RECONSTRUCTION RESULTS (12-Lead ECG)")
    print("=" * 50)
    print(f"PCC:     {pcc:.4f}")
    print(f"MAE:     {mae:.4f}")
    print(f"RMSE:    {rmse:.4f}")
    print(f"R2:      {r2:.4f}")
    print(f"perplexity:      {mean_perplexity:.4f}")

    # === Save Results ===
    result_dir = cfg["training"]["result"]
    os.makedirs(result_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(result_dir, "test_multitask_results.txt"), "w") as f:
        f.write("=== Atrial Fibrillation Detection ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (Macro): {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("\n" + report_str)
        f.write("\n\n=== ECG Reconstruction ===\n")
        f.write(f"PCC:     {pcc:.4f}\n")
        f.write(f"MAE:     {mae:.4f}\n")
        f.write(f"RMSE:    {rmse:.4f}\n")
        f.write(f"R2:      {r2:.4f}\n")
        f.write(f"perplexity:      {mean_perplexity:.4f}\n")

    # Save predictions CSV
    df = pd.DataFrame({
        "segment_id": results['ids'],
        "true_label": labels,
        "pred_label": preds,
        "af_probability": probs
    })
    df.to_csv(os.path.join(result_dir, "multitask_predictions.csv"), index=False)

    # Save visualizations
    for i, data in enumerate(results['visualize']):
        save_path = os.path.join(result_dir, f"recon_{i}_{data['id']}_label{data['label']}.png")
        plot_ecg_comparison(
            data['target'], data['recon'], data['id'], data['label'], save_path,
            fs=cfg["augmentation"].get("fs", 250)
        )
    print(f"\n✅ Saved {len(results['visualize'])} ECG plots to {result_dir}")
    print(f"✅ All results saved in {result_dir}")


if __name__ == "__main__":
    main()