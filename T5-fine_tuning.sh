#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J WikiSQL
#SBATCH -o WikiSQL.%J.out
#SBATCH -e WikiSQL.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=06:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=v100&local_200G

#run the application:
cd /home/toibazd/Data/Text2SQL/
pip install -q datasets, rouge_score
python -u T5-fine_tuning_with_WikiSQL.py

