#!/bin/bash
#SBATCH --job-name=freevc
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc

cd "$EXP"

echo "================================================"
echo "  FreeVC — Zero-shot Eval (pretrained, no tuning)"
echo "================================================"
python scripts/run_eval_zeroshot.py

echo ""
echo "================================================"
echo "  FreeVC — Fine-tuning on CUCO Tonsill"
echo "================================================"
python scripts/finetune.py \
    --surgery Tonsill \
    --epochs 200 \
    --batch_size 4 \
    --lr 2e-5

echo ""
echo "================================================"
echo "  FreeVC — Eval after fine-tuning"
echo "================================================"
python scripts/run_eval_zeroshot.py \
    --ckpt "$EXP/checkpoints/freevc_finetuned.pth" \
    --out_dir "$EXP/converted_test_finetuned"

echo "Done!"
