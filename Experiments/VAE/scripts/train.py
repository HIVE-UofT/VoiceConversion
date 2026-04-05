import torch
import torch.nn.functional as F
from model.model import SurgeryVAE
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

os.makedirs('plots', exist_ok=True)

# -----------------------------------------------------------------------
# Regularization hyper-parameters
# -----------------------------------------------------------------------
MIXUP_ALPHA = 0.2    # Beta distribution alpha for Manifold Mixup
L2_WEIGHT   = 1e-3   # Weight on ||mu_c||^2 + ||mu_s||^2 L2 penalty


class SurgeryDataset(Dataset):
    def __init__(self, pkl_path, target_len=400):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        mel   = item['mel_spectrogram']
        label = item['label']
        meta  = item.get('metadata', np.zeros(13, dtype=np.float32))

        if mel.shape[1] > self.target_len:
            mel = mel[:, :self.target_len]
        elif mel.shape[1] < self.target_len:
            pad_amount = self.target_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_amount)), mode='constant')

        return (
            torch.from_numpy(mel).float().unsqueeze(0),
            torch.tensor([label]).float(),
            torch.from_numpy(meta).float(),
        )


def training_step(model, x, labels, meta, alpha, gamma,
                  beta_c=1.0, mixup_alpha=MIXUP_ALPHA, l2_weight=L2_WEIGHT):
    """
    Single forward + loss computation.

    Manifold Mixup:  mix encoder hidden states h (after self.conv) so the
                     model sees interpolated manifold points during training.
    L2 regularization: explicit ||mu_c||^2 + ||mu_s||^2 penalty on latent
                       means — keeps them anchored near the origin and
                       complements the KL term.
    VQ surgery latent: z_s is now a discrete codebook lookup; its commitment
                       loss replaces the old kl_s term.
    """
    # ---- 1. Run conv + metadata encoder once ----
    h        = model.conv(x)           # (B, 128, 10, T')
    meta_emb = model.meta_encoder(meta)  # (B, meta_hidden)

    # ---- 2. Manifold Mixup (training only) ----
    if model.training and mixup_alpha > 0:
        lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
        perm = torch.randperm(x.size(0), device=x.device)
        h    = lam * h + (1 - lam) * h[perm]
        labels_for_loss = lam * labels + (1 - lam) * labels[perm]
    else:
        lam = 1.0
        labels_for_loss = labels

    # ---- 3. Forward from mixed features ----
    recon_final, mu_c, var_c, mu_s, z_s_q, vq_loss, s_pred_adv, _ = \
        model.forward_from_features(h, meta_emb, alpha,
                                    target_size=(x.size(2), x.size(3)))

    # ---- 4. Losses ----

    # Reconstruction
    loss_recon = F.l1_loss(recon_final, x, reduction='mean')

    # KL on content latent (VAE stays continuous for content)
    kl_c = -0.5 * torch.mean(1 + var_c - mu_c.pow(2) - var_c.exp())

    # VQ commitment loss on surgery latent (replaces kl_s)
    # vq_loss already computed inside surgery_vq

    # Adversarial loss — uses mixed labels
    loss_adv = F.binary_cross_entropy(s_pred_adv, labels_for_loss)

    # Surgery truth loss — uses quantized z_s_q with mixed labels
    s_pred_truth      = torch.sigmoid(model.surgery_truth_classifier(z_s_q))
    loss_surgery_truth = F.binary_cross_entropy(s_pred_truth, labels_for_loss)

    # L2 regularization on latent means
    l2_loss = l2_weight * (mu_c.pow(2).mean() + mu_s.pow(2).mean())

    total_loss = (loss_recon
                  + beta_c * kl_c
                  + vq_loss
                  + loss_adv
                  + loss_surgery_truth
                  + l2_loss)

    return {
        "total":   total_loss,
        "recon":   loss_recon.item(),
        "kl_c":    kl_c.item(),
        "vq":      vq_loss.item(),
        "adv":     loss_adv.item(),
        "surgery": loss_surgery_truth.item(),
        "l2":      l2_loss.item(),
    }


# --- Main Execution ---
device    = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")
model     = SurgeryVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

DATA_ROOT = '/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data'
train_loader = DataLoader(SurgeryDataset(f'{DATA_ROOT}/train_dataset.pkl'), batch_size=16, shuffle=True)
val_loader   = DataLoader(SurgeryDataset(f'{DATA_ROOT}/val_dataset.pkl'),   batch_size=16)
test_loader  = DataLoader(SurgeryDataset(f'{DATA_ROOT}/test_dataset.pkl'),  batch_size=16)

print(f"Length of train_loader: {len(train_loader)}")
epochs      = 200
total_steps = len(train_loader) * epochs
global_step = 0

train_losses = []
val_losses   = []
adv_losses   = []
surgery_truth_losses = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    epoch_adv        = 0
    epoch_surgery    = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for x, labels, meta in pbar:
        x, labels, meta = x.to(device), labels.to(device), meta.to(device)

        alpha = min(1.0, global_step / (total_steps * 0.2))
        beta  = min(1.0, global_step / (total_steps * 0.4))
        gamma = max(0.1, 1.0 - 0.5 * (global_step / (total_steps * 0.5)))

        optimizer.zero_grad()
        loss_dict = training_step(model, x, labels, meta, alpha, gamma, beta_c=beta)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_adv        += loss_dict["adv"]
        epoch_surgery    += loss_dict["surgery"]
        epoch_train_loss += loss_dict["total"].item()
        pbar.set_postfix({
            "total": f'{loss_dict["total"].item():.3f}',
            "vq":    f'{loss_dict["vq"]:.3f}',
            "adv":   f'{loss_dict["adv"]:.3f}',
        })
        global_step += 1

    train_losses.append(epoch_train_loss / len(train_loader))
    adv_losses.append(epoch_adv / len(train_loader))
    surgery_truth_losses.append(epoch_surgery / len(train_loader))

    # --- Validation & Visualization ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (x_val, labels_val, meta_val) in enumerate(val_loader):
            x_val, labels_val, meta_val = x_val.to(device), labels_val.to(device), meta_val.to(device)
            v_loss = training_step(model, x_val, labels_val, meta_val,
                                   alpha=alpha, gamma=gamma, beta_c=beta)['total']
            val_loss += v_loss.item()

            if i == 3:
                recon, *_ = model(x_val, meta_val, alpha=alpha)
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1); plt.title("Original")
                plt.imshow(x_val[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                plt.subplot(1, 2, 2); plt.title("Reconstructed")
                plt.imshow(recon[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                plt.savefig(f"plots/epoch_{epoch}_recon.png")
                plt.close()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)
    print(f"\nSummary Epoch {epoch} | Train: {train_losses[-1]:.4f} | Val: {avg_val_loss:.4f}")

# --- FINAL PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)
plt.savefig('plots/loss_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), adv_losses, label='Adversarial Loss (Content Latent)', color='red')
plt.plot(range(epochs), surgery_truth_losses, label='Surgery Truth Loss (Surgery Latent)', color='green')
plt.title('Classifier Performance: Disentanglement Tracking')
plt.xlabel('Epochs'); plt.ylabel('BCE Loss')
plt.legend(); plt.grid(True, linestyle='--')
plt.savefig('plots/classifier_losses.png')
plt.show()

# --- FINAL TESTING PHASE ---
model.eval()
test_loss = 0
with torch.no_grad():
    for x_test, labels_test, meta_test in tqdm(test_loader, desc="Testing"):
        x_test, labels_test, meta_test = x_test.to(device), labels_test.to(device), meta_test.to(device)
        t_loss = training_step(model, x_test, labels_test, meta_test,
                               alpha=alpha, gamma=gamma, beta_c=beta)
        test_loss += t_loss["total"].item()

avg_test_loss = test_loss / len(test_loader)
print(f"\nFinal Test Loss: {avg_test_loss:.4f}")
