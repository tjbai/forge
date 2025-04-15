#!/bin/bash
#SBATCH --job-name=sweep_pncg
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --array=1-4
#SBATCH --time=4:00:00
#SBATCH --output=sweep_pncg_%A_%a.out

uv run wandb agent --count=25 tjbai/mcmc/2h3qugp2
