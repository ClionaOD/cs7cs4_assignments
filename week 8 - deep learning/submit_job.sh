#!/bin/bash
#
#SBATCH --output=/home/clionaodoherty/cs7cs4_assignments/5K_baseline.out
#SBATCH --error=/home/clionaodoherty/cs7cs4_assignments/5K_baseline.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12


python /home/clionaodoherty/cs7cs4_assignments/week8.py