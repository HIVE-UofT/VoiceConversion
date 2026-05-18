#!/bin/bash
#SBATCH --job-name=dla_vc_v3_reeval
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

# Re-evaluate the DLA-VC v3 checkpoint (job 60816476 -> results_v3/best_model.pth)
# under the same conditions used originally: stock bshall/knn-vc HiFi-GAN,
# 5 held-out test patients. Should reproduce the +0.0450 mean test Delta.

export FORCE_STOCK_VOCODER=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lustre06/project/6086959/sepharfi/VoiceConversion/Experiments/dla_vc_v3

echo "================================================"
echo "  DLA-VC v3 - Re-Evaluation (stock HiFi-GAN)"
echo "  Checkpoint: results_v3/best_model.pth"
echo "  Original test Delta to reproduce: +0.0450"
echo "================================================"
python scripts/run_eval.py \
    --checkpoint results_v3/best_model.pth

echo "Done!"
