#!/bin/bash
#SBATCH --job-name=dla_vc_v3_multi
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

# DLA-VC v3 multi-surgery on top of dla_vc_v3:
#  + Trains a single DLA-VC on the combined train pools of Tonsill + Fess + Sept
#    (per-surgery held-out test patients excluded; ~15% stratified val split).
#  + KD teacher swapped to the multi-surgery UNet-VC-ECAPA model
#    (unet_vc_ecapa_multi/results_multi_v2/best_model.pt).
#  All other v3 settings retained: residual output, simplified losses
#  (cycle/content-cycle = 0), no audio aug, stock vocoder, ECAPA loss every 10
#  steps with lambda=1.0, KD lambda=5.0.

export FORCE_STOCK_VOCODER=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lustre06/project/6086959/sepharfi/VoiceConversion/Experiments/dla_vc_v3_multi

echo "================================================"
echo "  DLA-VC v3 Multi - Training (KD from UNet-VC-ECAPA-Multi + ECAPA loss)"
echo "================================================"
python scripts/train_split.py \
    --surgeries Tonsill,Fess,Sept \
    --output results_multi_v3 \
    --seed 42

echo ""
echo "================================================"
echo "  DLA-VC v3 Multi - Evaluation (stock HiFi-GAN, per-surgery + combined)"
echo "================================================"
python scripts/run_eval.py \
    --checkpoint results_multi_v3/best_model.pth

echo "Done!"
