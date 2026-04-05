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


# ---------------------------------------------------------------------------
# Vector Quantizer (adapted from vqvae/model/vqvae.py)
# Operates on (N, head_dim) flat feature vectors.
# ---------------------------------------------------------------------------
class VectorQuantizerHead(nn.Module):
    """
    Single VQ head with EMA codebook updates.
    Inputs:  (N, head_dim) flat features
    Outputs: (N, head_dim) quantized features, VQ loss, avg_probs
    """
    def __init__(self, num_codes=64, head_dim=8, commitment_weight=0.25,
                 ema_decay=0.95, dead_code_threshold=2, reset_every=50):
        super().__init__()
        self.num_codes = num_codes
        self.head_dim = head_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.reset_every = reset_every

        self.register_buffer('codebook', torch.randn(num_codes, head_dim))
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_sum', self.codebook.clone())
        self.register_buffer('usage_count', torch.zeros(num_codes))
        self.register_buffer('forward_count', torch.tensor(0, dtype=torch.long))
        self._initialized = False

    def _init_codebook(self, z_flat):
        if self._initialized:
            return
        n = z_flat.shape[0]
        if n >= self.num_codes:
            indices = torch.randperm(n, device=z_flat.device)[:self.num_codes]
            self.codebook.data.copy_(z_flat[indices])
        else:
            repeats = (self.num_codes // n) + 1
            self.codebook.data.copy_(z_flat.repeat(repeats, 1)[:self.num_codes])
        self.ema_sum.data.copy_(self.codebook.data.clone())
        self.ema_count.data.fill_(1.0)
        self._initialized = True

    def _reset_dead_codes(self, z_flat):
        dead_mask = self.usage_count < self.dead_code_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return
        replace_indices = torch.randint(0, z_flat.shape[0], (n_dead,), device=z_flat.device)
        noise = torch.randn_like(z_flat[replace_indices]) * 0.02
        self.codebook.data[dead_mask] = z_flat[replace_indices].detach() + noise
        self.ema_sum.data[dead_mask] = self.codebook.data[dead_mask].clone()
        self.ema_count.data[dead_mask] = 1.0
        self.usage_count.zero_()

    def forward(self, z_flat):
        if self.training:
            self._init_codebook(z_flat)

        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)
        z_q = self.codebook[indices]

        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()
                count = one_hot.sum(dim=0)
                summed = one_hot.t() @ z_flat

                self.ema_count.mul_(self.ema_decay).add_(count, alpha=1 - self.ema_decay)
                self.ema_sum.mul_(self.ema_decay).add_(summed, alpha=1 - self.ema_decay)

                n = self.ema_count.sum()
                count_stable = (
                    (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                )
                self.codebook.data.copy_(self.ema_sum / count_stable.unsqueeze(1))

                self.usage_count += count
                self.forward_count += 1
                if self.forward_count % self.reset_every == 0:
                    self._reset_dead_codes(z_flat)

        commitment_loss = F.mse_loss(z_q.detach(), z_flat)
        z_q = z_flat + (z_q - z_flat).detach()   # straight-through estimator
        vq_loss = self.commitment_weight * commitment_loss

        with torch.no_grad():
            avg_probs = F.one_hot(indices, self.num_codes).float().mean(dim=0)

        return z_q, vq_loss, avg_probs


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SurgeryVAE(nn.Module):
    def __init__(self, content_dim=512, surgery_dim=8, surgery_codes=64,
                 meta_dim=13, meta_hidden=32):
        super(SurgeryVAE, self).__init__()

        # --- ENCODER ---
        # Input: (B, 1, 80, 400)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # (40, 200)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # (20, 100)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (10, 50)
            nn.InstanceNorm2d(128), nn.ReLU()
        )

        # Content Head
        self.content_rnn = nn.LSTM(128*10, 256, bidirectional=True, batch_first=True)
        self.c_mu    = nn.Linear(512, content_dim)
        self.c_logvar = nn.Linear(512, content_dim)

        # Metadata Encoder: raw clinical features → embedding
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden),
            nn.ReLU(),
            nn.Linear(meta_hidden, meta_hidden)
        )

        # Surgery Head — conditioned on metadata, outputs pre-VQ embedding
        self.s_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.s_proj = nn.Linear(128 + meta_hidden, surgery_dim)   # pre-VQ projection

        # Vector Quantizer for surgery latent (replaces reparameterize on z_s)
        self.surgery_vq = VectorQuantizerHead(
            num_codes=surgery_codes,
            head_dim=surgery_dim,
            commitment_weight=0.25,
            ema_decay=0.95,
        )

        # Adversarial Detective (from Content)
        self.classifier = nn.Sequential(
            nn.Linear(content_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # Surgery Truth Classifier (from quantized Surgery Latent)
        self.surgery_truth_classifier = nn.Linear(surgery_dim, 1)

        # --- DECODER ---
        # Takes z_c + z_s_q + meta_embedding
        self.dec_fc  = nn.Linear(content_dim + surgery_dim + meta_hidden, 128*10)
        self.dec_rnn = nn.LSTM(128*10, 512, bidirectional=True, batch_first=True)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, (3,3), stride=2, padding=1, output_padding=1)
        )

        # Post-net for residual learning
        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512), nn.Tanh(),
            nn.Conv1d(512, 80, kernel_size=5, padding=2)
        )

    # ------------------------------------------------------------------
    # forward_from_features: called with already-computed conv output h
    # and meta_emb.  Separating this out lets training_step apply
    # Manifold Mixup on h before calling this.
    # ------------------------------------------------------------------
    def forward_from_features(self, h, meta_emb, alpha=1.0, target_size=(80, 400)):
        """
        h           : (B, 128, 10, T') output of self.conv
        meta_emb    : (B, meta_hidden)  output of self.meta_encoder
        target_size : (H, W) of the original mel-spectrogram, needed to size-match
                      recon_initial before post_net (transposed convs produce H=8, not 80)
        Returns     : recon_final, mu_c, var_c, mu_s, z_s_q, vq_loss, s_pred_adv, recon_initial
        """
        # Content branch
        c_in = h.permute(0, 3, 1, 2).flatten(2)   # (B, T', 1280)
        c_out, _ = self.content_rnn(c_in)
        mu_c  = self.c_mu(c_out)
        var_c = self.c_logvar(c_out)
        z_c   = self._reparameterize(mu_c, var_c)

        # Surgery branch — project then VQ-quantize
        s_in      = self.s_pool(h).flatten(1)                   # (B, 128)
        s_in_cond = torch.cat([s_in, meta_emb], dim=-1)         # (B, 128 + meta_hidden)
        mu_s      = self.s_proj(s_in_cond)                      # (B, surgery_dim)
        z_s_q, vq_loss, _ = self.surgery_vq(mu_s)               # (B, surgery_dim)

        # Adversarial step
        z_c_flat   = z_c.mean(dim=1)
        s_pred_adv = self.classifier(GRL.apply(z_c_flat, alpha))

        # Decode: tile z_s_q and meta_emb across time
        T          = z_c.size(1)
        z_s_tiled  = z_s_q.unsqueeze(1).expand(-1, T, -1)       # (B, T, surgery_dim)
        meta_tiled = meta_emb.unsqueeze(1).expand(-1, T, -1)    # (B, T, meta_hidden)
        z_joint    = torch.cat([z_c, z_s_tiled, meta_tiled], dim=-1)

        d_out, _      = self.dec_rnn(self.dec_fc(z_joint))
        d_out         = d_out.permute(0, 2, 1).unsqueeze(2)
        recon_initial = self.upsample(d_out)

        # The transposed convs produce (B,1,8,T) rather than (B,1,80,T).
        # Resize to target_size before post_net, which expects 80 mel channels.
        if recon_initial.shape[2:] != torch.Size(list(target_size)):
            recon_initial = F.interpolate(recon_initial, size=target_size,
                                          mode='bilinear', align_corners=False)

        residual    = self.post_net(recon_initial.squeeze(1))
        recon_final = recon_initial + residual.unsqueeze(1)

        return recon_final, mu_c, var_c, mu_s, z_s_q, vq_loss, s_pred_adv, recon_initial

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, meta, alpha=1.0):
        meta_emb = self.meta_encoder(meta)
        h        = self.conv(x)
        return self.forward_from_features(h, meta_emb, alpha,
                                          target_size=(x.size(2), x.size(3)))
