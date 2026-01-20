import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T 
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2ForCTC

class VAEMultiLoss(nn.Module):
    def __init__(self, kl_weight=1.0, ppg_weight=1.0, ecapa_weight=1.0, device="cuda"):
        super(VAEMultiLoss, self).__init__()
        self.kl_weight = kl_weight
        self.ppg_weight = ppg_weight
        self.ecapa_weight = ecapa_weight
        self.device = device
        
        # Load Frozen Feature Extractors
        self.ppg_extractor = load_ppg_extractor(device)
        self.ecapa_model = load_ecapa_tdnn(device)
        
        # Differentiable bridge: Matches your Librosa config (n_fft=1024, n_mels=128)
        self.inverse_mel = T.InverseMelScale(n_stft=513, n_mels=128, sample_rate=16000).to(device)
        self.griffin_lim = T.GriffinLim(n_fft=1024, hop_length=512).to(device)

        self.mse = nn.MSELoss() 
        self.l1 = nn.L1Loss()

    def forward(self, x_recon, x_orig, mu, logvar, wav_pre, wav_post):
        # 1. Reconstruction Loss (Mel-to-Mel)
        loss_recon = self.l1(x_recon, x_orig)

        # 2. KL Divergence
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_orig.size(0)

        # --- Differentiable Reconstruction ---
        # Inverse power_to_db (dB -> Power)
        x_recon_linear = torch.pow(10.0, x_recon / 10.0)
        wav_recon = self.griffin_lim(self.inverse_mel(x_recon_linear))
        
        # Ensure shape is [Batch, Time] for the models
        if wav_recon.ndim == 3:
            wav_recon = wav_recon.squeeze(1)

        # 3. ECAPA_TDNN Consistency Loss
        with torch.no_grad():
            ecapa_post = self.ecapa_model.encode_batch(wav_post)
        
        ecapa_recon = self.ecapa_model.encode_batch(wav_recon)
        # Squeeze to [Batch, 192] to ensure MSE is calculated correctly
        loss_ecapa = self.mse(ecapa_recon.squeeze(), ecapa_post.squeeze())

        # 4. PPG Loss
        with torch.no_grad():
            ppg_pre = self.ppg_extractor(wav_pre).logits
        
        ppg_recon = self.ppg_extractor(wav_recon).logits
        
        # Temporal alignment check (Wav2Vec2 is sensitive to exact signal length)
        if ppg_recon.size(1) != ppg_pre.size(1):
            ppg_recon = F.interpolate(ppg_recon.transpose(1, 2), size=ppg_pre.size(1)).transpose(1, 2)
            
        loss_ppg = self.mse(ppg_recon, ppg_pre)

        # Total Weighted Loss
        total_loss = (loss_recon + 
                      (self.kl_weight * loss_kl) + 
                      (self.ppg_weight * loss_ppg) + 
                      (self.ecapa_weight * loss_ecapa))

        return {
            "total_loss": total_loss,
            "recon": loss_recon,
            "kl": loss_kl,
            "ppg": loss_ppg,
            "ecapa": loss_ecapa
        }

def load_ecapa_tdnn(device="cuda"):
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    for param in model.parameters(): param.requires_grad = False
    model.eval()
    return model

def load_ppg_extractor(device="cuda"):
    model = Wav2Vec2ForCTC.from_pretrained("speech31/wav2vec2-large-english-TIMIT-phoneme_v3").to(device)
    for param in model.parameters(): param.requires_grad = False
    model.eval()
    return model


class MyVAE(nn.Module):
    def __init__(self, n_mels=128, latent_dim=128):
        super(MyVAE, self).__init__()
        
        # 1. Encoder
        self.encoder = nn.Sequential(
            # Layer 1: [B, 1, 128, 157] -> [B, 32, 64, 79]
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 2: [B, 32, 64, 79] -> [B, 64, 32, 40]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: [B, 64, 32, 40] -> [B, 128, 16, 20]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Latent Space (Flattened size: 128 * 16 * 20 = 40960)
        self.fc_mu = nn.Linear(40960, latent_dim)
        self.fc_var = nn.Linear(40960, latent_dim)
        
        # 2. Decoder: Upsamples Latent -> [1, 128, 157]
        self.decoder_input = nn.Linear(latent_dim, 40960)
        
        self.decoder = nn.Sequential(
            # Layer 1: [B, 128, 16, 20] -> [B, 64, 32, 40]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 2: [B, 64, 32, 40] -> [B, 32, 64, 80]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 3: [B, 32, 64, 80] -> [B, 1, 128, 160] (roughly)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
            # Final output is dB scale; no BatchNorm or ReLU here
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: [Batch, 1, 128, 157]
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)
        z = self.reparameterize(mu, logvar)
        
        # Reshape for decoder: [Batch, 128, 16, 20]
        z_dec = self.decoder_input(z).view(-1, 128, 16, 20)
        recon_x = self.decoder(z_dec)
        
        # Final Crop: The TransposeConv might produce 159 frames due to padding.
        # We crop it to exactly 157 to match x_orig.
        return recon_x[:, :, :, :157], mu, logvar