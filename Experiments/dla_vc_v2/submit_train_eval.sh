#!/bin/bash
#SBATCH --job-name=dla_vc_v2
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

# v2 architectural fixes:
#  1. Residual-output mode ON: output = pre_features + alpha * decoder_delta
#  2. Decoder skip connections NOT instance-normed (raw skips preserve magnitude)
#  3. WARMUP_EPOCHS=0; conv_loss active from epoch 1 to keep alpha non-zero
#  4. Loss simplification: LAMBDA_CYCLE=0, LAMBDA_CONTENT_CYCLE=0,
#     LAMBDA_CONV bumped 5->10, LAMBDA_RECON dropped 10->5, LAMBDA_Q_SHIFT 2->1
#  5. NO audio augmentation
#  6. Stock kNN-VC HiFi-GAN at eval (fine-tuned vocoder over-fit on 23 patients)

export FORCE_STOCK_VOCODER=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/dla_vc_v2

echo "================================================"
echo "  DLA-VC v2 - Training"
echo "================================================"
python scripts/train_split.py \
    --surgery Tonsill \
    --output results_v2 \
    --seed 42

echo ""
echo "================================================"
echo "  DLA-VC v2 - Evaluation (stock HiFi-GAN)"
echo "================================================"
python scripts/run_eval.py \
    --checkpoint results_v2/best_model.pth

echo "Done!"
