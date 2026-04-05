#!/bin/bash
#SBATCH --job-name=unet_v2_loo
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

module load python/3.10

source ../.env/bin/activate

echo "=========================================="
echo "  UNet-VC v2 — LOO with improvements"
echo "=========================================="

# ═══ Tonsill: Cross-patient pairing ═══
echo ""
echo "--- Tonsill (cross-patient) ---"
python scripts/train_loo.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2 \
    --output results_tonsill_cross \
    --cross_patient

# ═══ Tonsill: Same-patient pairing (ablation) ═══
echo ""
echo "--- Tonsill (same-patient, ablation) ---"
python scripts/train_loo.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2 \
    --output results_tonsill_same \
    --same_patient

# ═══ Fess ═══
echo ""
echo "--- Fess (cross-patient) ---"
python scripts/train_loo.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/2 \
    --output results_fess_cross \
    --cross_patient

# ═══ Sept ═══
echo ""
echo "--- Sept (cross-patient) ---"
python scripts/train_loo.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Sept/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Sept/Speech/2 \
    --output results_sept_cross \
    --cross_patient

echo ""
echo "=========================================="
echo "  All done!"
echo "=========================================="
