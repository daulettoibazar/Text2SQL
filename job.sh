#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J Evaluation_FT
#SBATCH -o Evaluation_FT.%J.out
#SBATCH -e Evaluation_FT.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=v100&local_200G

#run the application:
module load cudnn/8.8.1-cuda11.8.0
source activate base
cd /home/toibazd/Data/Text2SQL/
python testing_evaluation.py --model_name T5-Finetuned-with-IMDB-Spider \
--tokenizer_path T5-Finetuned-with-IMDB-Spider \
--dataset_path Training_set_IMDB/testing_set_no_target.csv \
--input_max 128 --output_max 256 

