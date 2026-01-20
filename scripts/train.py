import pandas as pd
from datasets import load_from_disk
import librosa
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import MyVAE, VAEMultiLoss
import torch.nn.functional as F
import torch.nn as nn   
import os
from tqdm import tqdm


ds_segmented = load_from_disk("/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/segmented_dataset")
# 1. Get all unique original file IDs (removing the '_segX' suffix)
all_ids = list(set([fid.split("_seg")[0] for fid in ds_segmented["file_id"]]))

# 2. Split IDs into Train (80%), Val (10%), Test (10%)
train_ids, temp_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# 3. Filter the segmented dataset based on these splits
train_ds = ds_segmented.filter(lambda x: x["file_id"].split("_seg")[0] in train_ids)
val_ds = ds_segmented.filter(lambda x: x["file_id"].split("_seg")[0] in val_ids)
test_ds = ds_segmented.filter(lambda x: x["file_id"].split("_seg")[0] in test_ids)
# Set format to return Tensors for the columns used in training
cols_to_tensor = ["mel_pre", "mel_post", "wav_pre", "wav_post"]
train_ds.set_format(type="torch", columns=cols_to_tensor, output_all_columns=True)
val_ds.set_format(type="torch", columns=cols_to_tensor, output_all_columns=True)
test_ds.set_format(type="torch", columns=cols_to_tensor, output_all_columns=True)

def collate_fn(batch):
    # Mel spectrograms need a channel dimension for Conv2d: [Batch, 1, 128, 157]
    mel_pre = torch.stack([item["mel_pre"] for item in batch]).unsqueeze(1)
    mel_post = torch.stack([item["mel_post"] for item in batch]).unsqueeze(1)
    
    # Waveforms for loss models: [Batch, 80000]
    wav_pre = torch.stack([item["wav_pre"] for item in batch])
    wav_post = torch.stack([item["wav_post"] for item in batch])
    
    return {
        "mel_pre": mel_pre,
        "mel_post": mel_post,
        "wav_pre": wav_pre,
        "wav_post": wav_post,
        "surgery_type": [item["surgery_type"] for item in batch],
        "file_id": [item["file_id"] for item in batch]
    }

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)} | Test Batches: {len(test_loader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = MyVAE(n_mels=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = VAEMultiLoss(device=device)

epochs = 80
kl_weight_max = 1.0
best_val_loss = float('inf')

# Ensure directory for checkpoints exists
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    train_running_loss = 0.0
    
    # KL Annealing update
    current_kl_weight = min(kl_weight_max, epoch / 20.0) 
    criterion.kl_weight = current_kl_weight

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for batch in pbar:
        # Load batch to device
        pre_mel = batch['mel_pre'].to(device)
        post_mel = batch['mel_post'].to(device)
        pre_wav = batch['wav_pre'].to(device)
        post_wav = batch['wav_post'].to(device)

        # Forward Pass
        recon_mel, mu, logvar = model(pre_mel)

        # Calculate Multi-Loss
        loss_dict = criterion(recon_mel, post_mel, mu, logvar, pre_wav, post_wav)
        loss = loss_dict['total_loss']

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_train_loss = train_running_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    val_running_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            pre_mel = batch['mel_pre'].to(device)
            post_mel = batch['mel_post'].to(device)
            pre_wav = batch['wav_pre'].to(device)
            post_wav = batch['wav_post'].to(device)

            recon_mel, mu, logvar = model(pre_mel)
            loss_dict = criterion(recon_mel, post_mel, mu, logvar, pre_wav, post_wav)
            
            val_running_loss += loss_dict['total_loss'].item()

    avg_val_loss = val_running_loss / len(val_loader)
    
    print(f"\nSummary Epoch {epoch+1}:")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | KL Weight: {current_kl_weight:.2f}")

    # --- SAVE BEST MODEL ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join("checkpoints", "best_vae_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, checkpoint_path)
        print(f"⭐ New best model saved at {checkpoint_path}")

    print("-" * 30)
