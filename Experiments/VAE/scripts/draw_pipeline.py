"""
Generates a clean pipeline figure for the SurgeryVAE model.
Output: pipeline_figure.png in the VAE directory.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── colour palette ──────────────────────────────────────────────────────────
C_AUDIO    = '#2980B9'
C_CLINICAL = '#D35400'
C_CONTENT  = '#27AE60'
C_SURGERY  = '#8E44AD'
C_DECODER  = '#C0392B'
C_ADV      = '#7F8C8D'
C_ARROW    = '#2C3E50'


def box(ax, x, y, w, h, label, sublabel=None, color='#2980B9',
        fontsize=11, radius=0.3, text_color='white'):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.8, edgecolor='white',
                          facecolor=color, zorder=3, alpha=0.93)
    ax.add_patch(rect)
    cy = y + h / 2 + (0.18 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                ha='center', va='center', fontsize=fontsize - 2,
                color=text_color, alpha=0.9, zorder=4, style='italic')


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.0, label=None, lpos='top'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'), zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dy = 0.22 if lpos == 'top' else -0.28
        ax.text(mx, my + dy, label, ha='center', va='center',
                fontsize=8.5, color=color, zorder=5,
                bbox=dict(fc='#F8F9FA', ec='none', pad=1))


def curved_arrow(ax, x1, y1, x2, y2, rad=0.3, color=C_ARROW, lw=2.0, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'), zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.2, my, label, ha='left', va='center',
                fontsize=8.5, color=color, zorder=5,
                bbox=dict(fc='#F8F9FA', ec='none', pad=1))


def band(ax, y, h, label, color, W=22):
    rect = FancyBboxPatch((0.1, y), W - 0.2, h,
                          boxstyle='round,pad=0.0',
                          linewidth=0, facecolor=color, alpha=0.07, zorder=0)
    ax.add_patch(rect)
    ax.text(W - 0.3, y + h / 2, label, ha='right', va='center',
            fontsize=9, color=color, alpha=0.65, fontstyle='italic', zorder=1)


# ── row y-positions (bottom of each row) ────────────────────────────────────
Y_IN   = 11.5
Y_ENC  = 9.3
Y_BR   = 6.8      # branch row (content / surgery heads)
Y_VQ   = 5.0      # reparameterize / VQ codebook
Y_ADV  = 3.2      # adversarial helpers
Y_DEC  = 1.8      # decoder
Y_POST = 0.5      # post-net + output (combined annotation)

BH = 0.95         # standard box height
BH2 = 0.80        # smaller boxes

# ── background bands ─────────────────────────────────────────────────────────
band(ax, Y_IN - 0.15,  BH + 0.3,        'Input',         '#2C3E50')
band(ax, Y_ENC - 0.15, BH + 0.3,        'Encoder',       C_AUDIO)
band(ax, Y_BR - 0.15,  (Y_DEC - Y_BR + BH + 0.3), 'Latent Space', '#6C3483')
band(ax, Y_DEC - 0.15, BH + 0.3,        'Decoder',       C_DECODER)
band(ax, Y_POST - 0.1, BH2 + 0.2,       'Output',        C_AUDIO)

# ── INPUTS ───────────────────────────────────────────────────────────────────
box(ax, 1.5,  Y_IN,  4.5, BH, 'Mel-Spectrogram',  '(B, 1, 80, 400)', C_AUDIO,    fontsize=12)
box(ax, 15.5, Y_IN,  4.5, BH, 'Clinical Features', '(B, 13)',         C_CLINICAL, fontsize=12)

# ── CONV ENCODER ─────────────────────────────────────────────────────────────
box(ax, 3.5, Y_ENC, 7.5, BH,
    'Conv Encoder',
    '3 × Conv2d (stride 2)  →  (B, 128, 10, 50)',
    C_AUDIO, fontsize=12)

# ── METADATA MLP ─────────────────────────────────────────────────────────────
box(ax, 15.5, Y_ENC, 4.5, BH,
    'Metadata Encoder',
    'Linear 13→32→32  (ReLU)',
    C_CLINICAL, fontsize=12)

# ── CONTENT BRANCH ───────────────────────────────────────────────────────────
box(ax, 1.0, Y_BR, 5.0, BH,
    'Content Head',
    'Bi-LSTM  →  μ_c, log σ²_c  (512-d)',
    C_CONTENT, fontsize=11)

box(ax, 1.0, Y_VQ, 5.0, BH,
    'Reparameterize',
    'z_c ~ N(μ_c, σ²_c)',
    C_CONTENT, fontsize=11)

# ── SURGERY BRANCH ───────────────────────────────────────────────────────────
box(ax, 9.5, Y_BR, 6.0, BH,
    'Surgery Head',
    'AvgPool → cat(audio, meta_emb) → Linear  →  (B, 8)',
    C_SURGERY, fontsize=11)

box(ax, 9.5, Y_VQ, 6.0, BH,
    'VQ Codebook',
    '64 codes × 8-dim  (EMA updates, dead-code reset)',
    C_SURGERY, fontsize=11)

# ── ADVERSARIAL HELPERS ──────────────────────────────────────────────────────
box(ax, 0.5, Y_ADV, 4.5, BH2,
    'GRL + Content Classifier',
    'content latent  ↛  surgery label',
    C_ADV, fontsize=10)

box(ax, 10.0, Y_ADV, 5.0, BH2,
    'Surgery Classifier',
    'z_s_q  →  surgery label  ✓',
    C_ADV, fontsize=10)

# ── DECODER ──────────────────────────────────────────────────────────────────
box(ax, 4.0, Y_DEC, 11.5, BH,
    'Decoder  →  Post-Net',
    'cat(z_c, z_s_q, meta_emb) → Bi-LSTM → TransposedConv×3 → Conv1d residual×3',
    C_DECODER, fontsize=11)

# ── OUTPUT ───────────────────────────────────────────────────────────────────
box(ax, 7.0, Y_POST, 7.5, BH2,
    'Reconstructed Mel-Spectrogram',
    '(B, 1, 80, 400)',
    C_AUDIO, fontsize=12)

# ── ARROWS ───────────────────────────────────────────────────────────────────
# input mel → conv encoder
arrow(ax, 3.75, Y_IN, 6.0, Y_ENC + BH, C_AUDIO, label='audio')

# clinical → metadata MLP
arrow(ax, 17.75, Y_IN, 17.75, Y_ENC + BH, C_CLINICAL, label='13 features')

# conv encoder → content head
arrow(ax, 5.5, Y_ENC, 3.5, Y_BR + BH, C_CONTENT, label='h')

# conv encoder → surgery head
arrow(ax, 8.5, Y_ENC, 11.5, Y_BR + BH, C_SURGERY, label='h')

# meta MLP → surgery head
arrow(ax, 15.5, Y_ENC + BH / 2, 15.5, Y_BR + BH / 2, C_CLINICAL)
arrow(ax, 15.5, Y_BR + BH / 2, 15.5, Y_BR + BH / 2, C_CLINICAL)
# horizontal segment
ax.annotate('', xy=(15.5, Y_BR + BH / 2), xytext=(17.75, Y_ENC + BH / 2),
            arrowprops=dict(arrowstyle='->', color=C_CLINICAL, lw=2.0,
                            connectionstyle='arc3,rad=0.0'), zorder=2)
ax.text(16.6, Y_BR + BH / 2 + 0.22, 'meta_emb', ha='center', va='center',
        fontsize=8.5, color=C_CLINICAL,
        bbox=dict(fc='#F8F9FA', ec='none', pad=1))

# meta MLP → decoder (curved around the right side)
curved_arrow(ax, 17.75, Y_ENC, 15.0, Y_DEC + BH,
             rad=-0.2, color=C_CLINICAL, label='meta_emb\n(tiled)')

# content head → reparameterize
arrow(ax, 3.5, Y_BR, 3.5, Y_VQ + BH, C_CONTENT)

# reparameterize → decoder
arrow(ax, 3.5, Y_VQ, 6.0, Y_DEC + BH, C_CONTENT, label='z_c')

# surgery head → VQ codebook
arrow(ax, 12.5, Y_BR, 12.5, Y_VQ + BH, C_SURGERY)

# VQ codebook → decoder
arrow(ax, 12.5, Y_VQ, 11.5, Y_DEC + BH, C_SURGERY, label='z_s_q')

# content reparameterize → GRL
arrow(ax, 2.0, Y_VQ, 2.0, Y_ADV + BH2, C_ADV)

# VQ → surgery classifier
arrow(ax, 12.5, Y_VQ, 12.5, Y_ADV + BH2, C_ADV)

# decoder → output
arrow(ax, 9.75, Y_DEC, 10.75, Y_POST + BH2, C_DECODER)

# ── LOSS LABELS ──────────────────────────────────────────────────────────────
lstyle = dict(fontsize=9, color='#444', ha='center', va='center',
              bbox=dict(fc='#FFF8DC', ec='#CCA300', pad=4,
                        boxstyle='round,pad=0.4'))

ax.text(2.75, Y_ADV - 0.55,
        'L_adv\n(content ↛ surgery)', **lstyle)

ax.text(12.5, Y_ADV - 0.55,
        'L_surgery_truth\n(z_s_q → surgery?)', **lstyle)

ax.text(10.75, Y_POST - 0.45,
        'L_recon (L1)  +  β · L_KL  +  L_VQ  +  L_L2',
        fontsize=10, color='#333', ha='center', va='center',
        bbox=dict(fc='#FFF3CD', ec='#F0C040', pad=5,
                  boxstyle='round,pad=0.4'))

# ── TITLE ────────────────────────────────────────────────────────────────────
ax.set_title('SurgeryVAE — Disentangled Voice Conversion Pipeline\n'
             'for Post-Tonsillectomy Speech  (HAD5016, Winter 2026)',
             fontsize=15, fontweight='bold', color='#1A1A2E', pad=14)

# ── LEGEND ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_AUDIO,    label='Audio path'),
    mpatches.Patch(color=C_CLINICAL, label='Clinical metadata path'),
    mpatches.Patch(color=C_CONTENT,  label='Content branch  (continuous VAE)'),
    mpatches.Patch(color=C_SURGERY,  label='Surgery branch  (VQ discrete)'),
    mpatches.Patch(color=C_DECODER,  label='Decoder + Post-Net'),
    mpatches.Patch(color=C_ADV,      label='Adversarial components'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor='#CCC',
          bbox_to_anchor=(1.0, 0.01))

plt.tight_layout()
out = '/home/sepharfi/projects/def-zshakeri/sepehr/VoiceConversion/VAE/pipeline_figure.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#F8F9FA')
print(f'Saved → {out}')
