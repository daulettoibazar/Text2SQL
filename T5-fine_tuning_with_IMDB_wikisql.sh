#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJobLORA_T5_IMDB_wikisql
#SBATCH -o MyJobLORA_T5_IMDB_wikisql.%J.out
#SBATCH -e MyJobLORA_T5_IMDB_wikisql.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=v100

#run the application:
cd /home/toibazd/Data/Text2SQL/ 
module load cuda/11.4.4
python -u T5-fine_tuning_with_IMDB_wikisql.py

