import torch
import torch.nn as nn
import torch.nn.functional as F

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class SurgeryVAE(nn.Module):
    def __init__(self, content_dim=512, surgery_dim=8):
        super(SurgeryVAE, self).__init__()
        
        # --- ENCODER ---
        # Input: (B, 1, 80, 400)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # (40, 200)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (20, 100)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # (10, 50)
            nn.InstanceNorm2d(128), nn.ReLU() 
        )
        
        # Content Head
        self.content_rnn = nn.LSTM(128*10, 256, bidirectional=True, batch_first=True)
        self.c_mu = nn.Linear(512, content_dim)
        self.c_logvar = nn.Linear(512, content_dim)
        
        # Surgery Head
        self.s_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.s_mu = nn.Linear(128, surgery_dim)
        self.s_logvar = nn.Linear(128, surgery_dim)
        
        # Adversarial Detective (from Content)
        self.classifier = nn.Sequential(
            nn.Linear(content_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        
        # Surgery Truth Classifier (from Surgery Latent)
        self.surgery_truth_classifier = nn.Linear(surgery_dim, 1)
        
        # --- DECODER ---
        self.dec_fc = nn.Linear(content_dim + surgery_dim, 128*10)
        self.dec_rnn = nn.LSTM(128*10, 512, bidirectional=True, batch_first=True)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (3,3), stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (3,3), stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, (3,3), stride=2, padding=1, output_padding=1)     
        )

    def encode(self, x):
        h = self.conv(x)
        # Content branch
        c_in = h.permute(0, 3, 1, 2).flatten(2) 
        c_out, _ = self.content_rnn(c_in)
        mu_c, var_c = self.c_mu(c_out), self.c_logvar(c_out)
        # Surgery branch
        s_in = self.s_pool(h).flatten(1)
        mu_s, var_s = self.s_mu(s_in), self.s_logvar(s_in)
        return mu_c, var_c, mu_s, var_s

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, alpha=1.0):
        mu_c, var_c, mu_s, var_s = self.encode(x)
        z_c = self.reparameterize(mu_c, var_c)
        z_s = self.reparameterize(mu_s, var_s)
        
        # Adversarial step
        z_c_flat = z_c.mean(dim=1) 
        s_pred_adv = self.classifier(GRL.apply(z_c_flat, alpha))
        
        # Combine and decode
        z_s_tiled = z_s.unsqueeze(1).expand(-1, z_c.size(1), -1) 
        z_joint = torch.cat([z_c, z_s_tiled], dim=-1)
        d_out, _ = self.dec_rnn(self.dec_fc(z_joint))
        
        d_out = d_out.permute(0, 2, 1).unsqueeze(2) 
        recon = self.upsample(d_out)

        # FINAL FIX: Force recon size to match input x size exactly
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return recon, mu_c, var_c, mu_s, var_s, s_pred_adv