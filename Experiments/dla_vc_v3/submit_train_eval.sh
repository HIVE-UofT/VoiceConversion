#!/bin/bash
#SBATCH --job-name=dla_vc_v3
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

# v3 on top of v2:
#  + Knowledge distillation from frozen UNet-VC-ECAPA teacher (+0.0635 leader)
#    MSE + cos between DLA's x_a2b/x_b2a and teacher's pre-to-post conversion.
#  + Re-enabled ECAPA differentiable loss (every 10 steps, lambda=1.0 to keep noise bounded).
#  All other v2 settings retained: residual output, no instance-norm on skips,
#  simplified losses (cycle/content-cycle = 0), no audio aug, stock vocoder.

export FORCE_STOCK_VOCODER=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/dla_vc_v3

echo "================================================"
echo "  DLA-VC v3 - Training (KD from UNet-VC-ECAPA + ECAPA loss)"
echo "================================================"
python scripts/train_split.py \
    --surgery Tonsill \
    --output results_v3b \
    --seed 42

echo ""
echo "================================================"
echo "  DLA-VC v3 - Evaluation (stock HiFi-GAN)"
echo "================================================"
python scripts/run_eval.py \
    --checkpoint results_v3b/best_model.pth

echo "Done!"
