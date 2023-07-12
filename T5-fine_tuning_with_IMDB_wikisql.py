#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[4]:





# In[9]:





# In[ ]:


from pynvml import *
import torch.optim as optim


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    


# In[ ]:


print_gpu_utilization()


# In[ ]:


# !pip -q datasets


# In[10]:


model_name = "t5-small"
from transformers import AutoTokenizer, T5ForConditionalGeneration


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# In[ ]:


from datasets import load_dataset


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset




# In[ ]:


df1 = pd.read_csv('Training_set_IMDB/training_set_sample_no_target.csv')
df1 = df1.dropna()


shuffled_df = df1.sample(frac=1, random_state=42)  # Set random_state for reproducibility

train_df, eval_df = train_test_split(shuffled_df, test_size=0.1, random_state=42)

# 'train_df' will contain 80% of the shuffled data for training
# 'eval_df' will contain 20% of the shuffled data for evaluation


text = []
sql = []

for index, row in train_df.iterrows():
    
    text_input = "Translate to SQL: " + row['Text'],
    sql_input = row['SQL']
    text.append(text_input)
    sql.append(sql_input)
    
inputs = {"inputs": text,
        "target": sql}

train_dataset = Dataset.from_dict(inputs)


# In[ ]:


text_2 = []
sql_2 = []

for index, row in eval_df.iterrows():
    
    text_input = "Translate to SQL: " + row['Text'],
    sql_input = row['SQL']
    text_2.append(text_input)
    sql_2.append(sql_input)
    
inputs_2 = {"inputs": text_2,
        "target": sql_2}
eval_dataset = Dataset.from_dict(inputs_2)


# In[ ]:


def map_to_lenght(x):
    x["input_len"] = len(tokenizer(x["inputs"]).input_ids)
    x["input_longer_256"] = int(x["input_len"]>256)
    x["input_longer_128"] = int(x["input_len"]>128)
    x["input_longet_64"] = int(x["input_len"]>64)
    x["output_len"] = len(tokenizer(x["target"]).input_ids)
    x["output_longet_256"] = int(x["output_len"]>256)
    x["output_longet_128"] = int(x["output_len"]>128)
    x["output_longet_64"] = int(x["output_len"]>64)
    return x

sample_size = 10000
data_stats = train_dataset.select(range(sample_size)).map(map_to_lenght, num_proc=4)


# In[ ]:


def compute_and_print(x):
    if len(x["input_len"])==sample_size:
        print(
            f"Input mean: {sum(x['input_len'])/sample_size} \n % of input len > 256: {sum(x['input_longer_256'])/sample_size}, \n % of input len > 128: {sum(x['input_longer_128'])/sample_size}, \n % of input len > 64: {sum(x['input_longet_64'])/sample_size}, \n Ouput mean: {sum(x['output_len'])/sample_size},\n% of output len > 256: {sum(x['output_longet_256'])/sample_size}, \n% of output len > 128: {sum(x['output_longet_128'])/sample_size}, \n% of output len > 64: {sum(x['output_longet_64'])/sample_size}")

output = data_stats.map(compute_and_print, batched=True, batch_size=-1)


# In[ ]:


def convert_to_features(example_batch, padding = "max_length"):
    inputs = tokenizer.batch_encode_plus(example_batch["inputs"],is_split_into_words=True, max_length=64, truncation=True)
    
    targets = tokenizer.batch_encode_plus(example_batch["target"], max_length=256,truncation = True)
    if padding == "max_length":
        targets["inputs_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in target] for target in targets["input_ids"]
        ]
    
    inputs["labels"] = targets['input_ids']
    return inputs

train_data = train_dataset.map(convert_to_features, batched=True, remove_columns=train_dataset.column_names)
test_data = eval_dataset.map(convert_to_features, batched=True, remove_columns=eval_dataset.column_names)


# In[ ]:


test_data


# In[ ]:


columns = ['input_ids', 'attention_mask', 'labels']

train_data.set_format(type='torch', columns=columns)
test_data.set_format(type='torch', columns=columns)


# In[ ]:


len(train_data)


# In[ ]:


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import os

output_dir = 'T5-fine-tuned-with-IMDB-wikisql'
os.mkdir(output_dir)


# In[ ]:


training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    learning_rate=2e-4,
    weight_decay=0.1,
    do_eval=True,
    logging_strategy="epoch",
    save_strategy="epoch",
    overwrite_output_dir=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=True 
)


# In[ ]:


get_ipython().system(' pip install -q rouge_score')


# In[ ]:


from evaluate import load
rouge = load("rouge")


# In[ ]:


import numpy as n

def compute_metrics(pred):
    predictions, labels = pred
    
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels[labels== -101] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    return {
        "rouge2 precision": round(rouge_output.precision, 4),
        "rouge2 recall": round(rouge_output.recall, 4),
        "rouge2 F1 score": round(rouge_output.fmeasure, 4)
    }


# In[ ]:


test_data

data_collator = DataCollatorForSeq2Seq(tokenizer)


# In[ ]:


trainer = Seq2SeqTrainer(
    model = model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset= test_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.save_model(output_dir)


# In[ ]:


tokenizer.save_pretrained(output_dir)


# In[ ]:





# In[ ]:




