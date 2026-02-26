import torch
import torch.nn.functional as F
from model.model import SurgeryVAE # Ensure your path is correct
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

os.makedirs('plots', exist_ok=True)

class SurgeryDataset(Dataset):
    def __init__(self, pkl_path, target_len=400):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.target_len = target_len
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        mel = item['mel_spectrogram'] 
        label = item['label'] 
        
        if mel.shape[1] > self.target_len:
            mel = mel[:, :self.target_len]
        elif mel.shape[1] < self.target_len:
            pad_amount = self.target_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_amount)), mode='constant')
            
        return torch.from_numpy(mel).float().unsqueeze(0), torch.tensor([label]).float()

def training_step(model, x, labels, alpha, beta_c=1.0, beta_s=0.1):
    recon_final, mu_c, var_c, mu_s, var_s, s_pred_adv, recon_initial = model(x, alpha)
    
    # 1. Recon Loss (Calculated on both for better gradient flow)
    loss_recon_init = F.l1_loss(recon_initial, x, reduction='mean')
    loss_recon_final = F.l1_loss(recon_final, x, reduction='mean')
    loss_recon = loss_recon_init + loss_recon_final
    
    # 2. KL Losses
    kl_c = -0.5 * torch.mean(1 + var_c - mu_c.pow(2) - var_c.exp())
    kl_s = -0.5 * torch.mean(1 + var_s - mu_s.pow(2) - var_s.exp())
    
    # 3. Adversarial Loss (from Content)
    loss_adv = F.binary_cross_entropy(s_pred_adv, labels)

    # 4. Surgery Truth Loss (from Surgery Latent)
    z_s = model.reparameterize(mu_s, var_s)
    s_pred_truth = torch.sigmoid(model.surgery_truth_classifier(z_s))
    loss_surgery_truth = F.binary_cross_entropy(s_pred_truth, labels)
    
    total_loss = loss_recon + (beta_c * kl_c) + (beta_s * kl_s) + loss_adv + loss_surgery_truth
    return total_loss

# --- Main Execution ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = SurgeryVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
# Loaders
train_loader = DataLoader(SurgeryDataset('/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data/train_dataset.pkl'), batch_size=16, shuffle=True)
val_loader = DataLoader(SurgeryDataset('/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data/val_dataset.pkl'), batch_size=16)


print(f"Length of train_loader: {len(train_loader)}")
epochs = 200
total_steps = len(train_loader) * epochs
global_step = 0

for epoch in range(epochs):
    model.train()
    # Using tqdm for a live progress bar in the terminal
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for x, labels in pbar:
        x, labels = x.to(device), labels.to(device)
        
        alpha = min(1.0, global_step / (total_steps * 0.2))
        beta = min(1.0, global_step / (total_steps * 0.4))
        
        optimizer.zero_grad()
        
        # Tip: Modify training_step to return a dictionary of losses
        loss = training_step(model, x, labels, alpha, beta_c=beta, beta_s=beta*0.1)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "alpha": f"{alpha:.2f}"})
        global_step += 1

    # --- Validation & Visualization ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (x_val, labels_val) in enumerate(val_loader):
            x_val, labels_val = x_val.to(device), labels_val.to(device)
            v_loss = training_step(model, x_val, labels_val, alpha=1.0, beta_c=1.0, beta_s=0.1)
            val_loss += v_loss.item()
            
            # VISUAL CHECK: Save the first batch's first sample reconstruction
            if i == 3:
                recon, _, _, _, _, _ = model(x_val, alpha=1.0)
                
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(x_val[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                
                plt.subplot(1, 2, 2)
                plt.title("Reconstructed")
                plt.imshow(recon[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                
                plt.savefig(f"plots/epoch_{epoch}_recon.png")
                plt.close()

    print(f"\nSummary Epoch {epoch} | Val Loss: {val_loss/len(val_loader):.4f}")