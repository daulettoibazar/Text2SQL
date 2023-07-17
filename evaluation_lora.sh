#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J Evaluation_lora
#SBATCH -o Evaluation_lora.%J.out
#SBATCH -e Evaluation_lora.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=v100&local_200G

#run the application
module load cuda/11.4.4
source activate base
cd /home/toibazd/Data/Text2SQL/
python -u Model_evaluation_LORA.py
