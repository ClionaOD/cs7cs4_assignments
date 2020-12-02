#!/bin/bash
#
#SBATCH --output=/home/clionaodoherty/cs7cs4_assignments/10K.out
#SBATCH --error=/home/clionaodoherty/cs7cs4_assignments/10K.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12


python /home/clionaodoherty/cs7cs4_assignments/week8.py