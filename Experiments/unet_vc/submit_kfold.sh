#!/bin/bash
#SBATCH --job-name=unet_kfold
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

module load python/3.10

source ../.env/bin/activate

# ═══ Tonsill: 5-fold CV with 5 held-out test patients ═══
echo "=========================================="
echo "  Tonsill — 5-fold CV + held-out test"
echo "=========================================="

python scripts/train_kfold.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2 \
    --output checkpoints_kfold_tonsill \
    --n_test 5 --k_folds 5 --seed 42

python scripts/inference_kfold.py \
    --checkpoint checkpoints_kfold_tonsill/best_model.pt \
    --output_dir converted_kfold_tonsill

python scripts/evaluate.py \
    --converted_dir converted_kfold_tonsill \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2 \
    --method_name "UNet-VC (Tonsill, test only)" \
    --skip_f0

# ═══ Fess: 5-fold CV with 5 held-out test patients ═══
echo ""
echo "=========================================="
echo "  Fess — 5-fold CV + held-out test"
echo "=========================================="

python scripts/train_kfold.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/2 \
    --output checkpoints_kfold_fess \
    --n_test 5 --k_folds 5 --seed 42

python scripts/inference_kfold.py \
    --checkpoint checkpoints_kfold_fess/best_model.pt \
    --output_dir converted_kfold_fess

python scripts/evaluate.py \
    --converted_dir converted_kfold_fess \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/2 \
    --method_name "UNet-VC (Fess, test only)" \
    --skip_f0

# ═══ Sept: 5-fold CV with 5 held-out test patients ═══
echo ""
echo "=========================================="
echo "  Sept — 5-fold CV + held-out test"
echo "=========================================="

python scripts/train_kfold.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Sept/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Sept/Speech/2 \
    --output checkpoints_kfold_sept \
    --n_test 5 --k_folds 5 --seed 42

python scripts/inference_kfold.py \
    --checkpoint checkpoints_kfold_sept/best_model.pt \
    --output_dir converted_kfold_sept

python scripts/evaluate.py \
    --converted_dir converted_kfold_sept \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Sept/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Sept/Speech/2 \
    --method_name "UNet-VC (Sept, test only)" \
    --skip_f0

echo ""
echo "=========================================="
echo "  All done!"
echo "=========================================="
