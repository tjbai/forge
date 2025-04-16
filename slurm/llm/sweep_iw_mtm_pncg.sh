#!/bin/bash
#SBATCH --job-name=sweep_iw_mtm_pncg
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --array=1-4
#SBATCH --time=8:00:00
#SBATCH --output=sweep_iw_mtm_pncg_%A_%a.out

uv run wandb agent --count=50 tjbai/mcmc/brlhcx7r
