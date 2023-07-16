# Text2SQL

1) Fine Tuning Google T5 model on Text2SQL task.
2) Originally T5 model was fine-tuned using the WikiSQL dataset.
3) Next, the model was adapted with LoRA on the IMDB database.
4) IMDB dataset - synthetic dataset created upon relational IMDB database using standard patterns and placeholders.

* LoRA_adaptation_T5_WikiSQL.ipynb - Lora adaptation scripts notebook for T5-WikiSQL model
* LoRA_adaptation_T5_WikiSQL.py - Lora adaptation python script for T5-WikiSQL model
* Model_evalution - notebook script to evaluate fine-tuned models: execution rate and execution accuracy
* README.md - guideline
* T5-fine_tuning_with_IMDB_wikisql.ipynb - full T5-wikisql model fine-tuning on IMDB dataset
* T5-fine_tuning_with_IMDB_wikisql.py - python script for full T5-wikisql model fine-tuning on IMDB dataset
* T5-fine_tuning_with_IMDB_wikisql.sh - IBEX shell script for full T5-wikisql model fine-tuning on IMDB dataset
* T5-fine_tuning_with_Spider.ipynb - notebook to fine-tune base T5 on Spider dataset
* T5-fine_tuning_with_WikiSQL.ipynb - notebook to fine-tune base T5 on WikiSQL dataset
* T5-fine_tuning_with_WikiSQL.py - python script to fine-tune base T5 on WikiSQL dataset
* T5-wikisql_fine_tuning.sh - IBEX shell script to fine-tune base T5 on WikiSQL dataset
* Testing_T5_WikiSQL.ipynb - notebook to make inferences on T5 fine-tuned on WikiSQL dataset
* WikiSQL.26104292.err/.out - Fine-tuning process files of base T5 on WikiSQL
* /content - repo to base T5-small model fine-tuned on WikiSQL
* /LoRA_adapted_T5 - repo to wikisql-T5-small model LoRA adapted to IMDB
* /T5-fine-tuned-with-IMDB-wikisql - repo to wikisql-T5-small fully fine-tuned on IMDB
* /Training_set_IMDB - repo containing training and testing datasets: train - csv file; test - dataset object;

