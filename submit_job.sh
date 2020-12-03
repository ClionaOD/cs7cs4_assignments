#!/bin/bash
#
#SBATCH --output=/home/clionaodoherty/cs7cs4_assignments/5K_0.01.out
#SBATCH --error=/home/clionaodoherty/cs7cs4_assignments/5K_0.01.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12


python /home/clionaodoherty/cs7cs4_assignments/week8.py