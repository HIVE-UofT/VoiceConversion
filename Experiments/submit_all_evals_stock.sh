#!/bin/bash
#SBATCH --job-name=all_evals_stock
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ~/envs/myenv/bin/activate

# Force every method that goes through load_finetuned_knnvc() to use the
# STOCK kNN-VC HiFi-GAN. This neutralises the over-fit CUCO-fine-tuned
# vocoder and lets us compare methods on the same vocoder footing.
export FORCE_STOCK_VOCODER=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EXP=/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments

echo "========================================================="
echo "  Re-evaluating all methods with STOCK bshall/knn-vc"
echo "  HiFi-GAN. Test patients: 0045, 0085, 0110, 0122, 0132."
echo "========================================================="

echo ""
echo "=== kNN-VC ==="
cd $EXP/knn_vc && python scripts/run_eval.py

echo ""
echo "=== Mean-Shift ==="
cd $EXP/mean_shift && python scripts/run_eval.py

echo ""
echo "=== MKL-VC ==="
cd $EXP/mkl_vc && python scripts/run_eval.py

echo ""
echo "=== LinearVC ==="
cd $EXP/linear_vc && python scripts/run_eval.py

echo ""
echo "Done with training-free methods. Trained-method retrains submitted separately."
