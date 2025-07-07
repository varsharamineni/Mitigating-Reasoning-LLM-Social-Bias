#!/bin/bash
#SBATCH --job-name=bbq_bias_eval
#SBATCH --output=fit_and_score.%j.out
#SBATCH --error=fit_and_score.%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=euhpc_d19_099

# Load modules if needed
module load python/3.12
module load cuda/11.8  # or your CUDA version

# Activate your virtual environment
source /leonardo/home/userexternal/vraminen/Mitigating-Reasoning-LLM-Social-Bias/.venv/bin/activate

# Run your Python script
python /leonardo/home/userexternal/vraminen/Mitigating-Reasoning-LLM-Social-Bias/BBQ/run_new_exp.py
