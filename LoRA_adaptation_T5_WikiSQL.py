#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install --upgrade bitsandbytes 


# In[2]:


# !pip install --upgrade git+https://github.com/huggingface/peft.git 
# !pip install --upgrade git+https://github.com/huggingface/transformers.git
# !pip install --upgrade git+https://github.com/huggingface/accelerate.git
# !pip install --upgrade datasets
# !pip install --upgrade loralib


# In[13]:



# In[14]:


import torch

torch.cuda.is_available()


# In[15]:




# In[16]:


import os
print(os.environ.get('PYTHONPATH'))


# In[19]:


import os
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import bitsandbytes as bnb
import triton


model_name = "/home/toibazd/Data/Text2SQL/content/t5-small-finetuned-wikisql"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[20]:


print(model)


# In[ ]:





# In[21]:


for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)
        
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# In[22]:


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)


# In[23]:


def get_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
get_trainable_params(model)


# In[24]:


from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType


config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    lora_dropout=0.05,
    task_type= TaskType.SEQ_2_SEQ_LM,
    bias = "none"
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)
get_trainable_params(model)


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset


# In[26]:


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


# In[27]:


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


# In[29]:


eval_dataset['inputs'][0]


# In[30]:


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


# In[31]:


def compute_and_print(x):
    if len(x["input_len"])==sample_size:
        print(
            f"Input mean: {sum(x['input_len'])/sample_size} \n % of input len > 256: {sum(x['input_longer_256'])/sample_size}, \n % of input len > 128: {sum(x['input_longer_128'])/sample_size}, \n % of input len > 64: {sum(x['input_longet_64'])/sample_size}, \n Ouput mean: {sum(x['output_len'])/sample_size},\n% of output len > 256: {sum(x['output_longet_256'])/sample_size}, \n% of output len > 128: {sum(x['output_longet_128'])/sample_size}, \n% of output len > 64: {sum(x['output_longet_64'])/sample_size}")

output = data_stats.map(compute_and_print, batched=True, batch_size=-1)


# In[34]:


train_dataset["target"][-500:-1]


# In[36]:


def convert_to_features(example_batch, padding = "max_length"):
    inputs = tokenizer.batch_encode_plus(example_batch["inputs"],is_split_into_words=True, max_length=64, padding="max_length", truncation=True)
    targets = tokenizer.batch_encode_plus(example_batch["target"], max_length=256, padding="max_length",truncation = True)
    if padding == "max_length":
        targets["inputs_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in target] for target in targets["input_ids"]
        ]
        
    encodings = {
        "input_ids":inputs["input_ids"],
        "attention_mask":inputs["attention_mask"],
        "labels":targets["input_ids"],
        "decoder_attention_mask":targets["attention_mask"]
    }
    return encodings

train_data = train_dataset.map(convert_to_features, batched=True, remove_columns=train_dataset.column_names)
test_data = eval_dataset.map(convert_to_features, batched=True, remove_columns=eval_dataset.column_names)


# In[37]:


columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']

train_data.set_format(type='torch', columns=columns)
test_data.set_format(type='torch', columns=columns)


# In[38]:


len(test_data["input_ids"])


# In[39]:


def sequence_check(list_of_seq):
    len_seq = len(list_of_seq[0])
    for seq in list_of_seq:
        if len(seq)!=len_seq:
            print(len(seq))
            return False
        
    return True


# In[40]:


sequence_check(train_data['input_ids'])


# In[41]:


len(train_data["input_ids"][12330])


# In[43]:


from transformers import Seq2SeqTrainingArguments
output_dir = '/home/toibazd/Data/Text2SQL/LoRA_adapted_T5'
args = Seq2SeqTrainingArguments(
        auto_find_batch_size=True,
        gradient_accumulation_steps = 4,
        learning_rate=6e-4, 
        warmup_steps=100,  
        save_strategy="no",
        load_best_model_at_end=True,
        num_train_epochs=10,
        eval_accumulation_steps=10,
        fp16=True,
        logging_steps=100, 
        output_dir=output_dir)


# In[44]:


get_ipython().system(' pip install -q rouge_score evaluate')


# In[45]:


from evaluate import load
rouge = load("rouge")


# In[46]:


def compute_metrics(pred):
    label_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids== -101] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    print_gpu_utilization()
    return {
        "rouge2 precision": round(rouge_output.precision, 4),
        "rouge2 recall": round(rouge_output.recall, 4),
        "rouge2 F1 score": round(rouge_output.fmeasure, 4)
    }


# In[47]:


from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

label_pad_token_id = -100

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                      model=model,
                                      label_pad_token_id=label_pad_token_id)

trainer = Seq2SeqTrainer(
    model = model,
    train_dataset = train_data,
    args = args,
    data_collator=data_collator
)

model.config.use_cache = False


# In[48]:


from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# In[ ]:


trainer.train()


# In[ ]:


trainer.save_model()
tokenizer.save_pretrained(output_dir)


# In[ ]:





# In[ ]:




