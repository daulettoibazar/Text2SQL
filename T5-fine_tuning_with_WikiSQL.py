#!/usr/bin/env python
# coding: utf-8
#this is the python code to fine-tune T5 model on WikiSQL dataset



# In[2]:


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
    


# In[3]:


print_gpu_utilization()



# In[5]:


model_name = "t5-small"
from transformers import AutoTokenizer, T5ForConditionalGeneration


# In[6]:


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# In[7]:


from datasets import load_dataset


# In[8]:


train_data = load_dataset("wikisql", split="train+validation")
test_data = load_dataset("wikisql", split = "test")



# In[9]:


train_data[0]


# In[10]:


def format_dataset(dataset_batch):
    return {"input":"translate to SQL: " + dataset_batch["question"],"target":dataset_batch["sql"]["human_readable"]}

train_data = train_data.map(format_dataset, remove_columns=train_data.column_names)


# In[11]:


train_data[-1]


# In[12]:


test_data = test_data.map(format_dataset,remove_columns=test_data.column_names)


# In[13]:


def map_to_lenght(x):
    x["input_len"] = len(tokenizer(x["input"]).input_ids)
    x["input_longer_256"] = int(x["input_len"]>256)
    x["input_longer_128"] = int(x["input_len"]>128)
    x["input_longet_64"] = int(x["input_len"]>64)
    x["output_len"] = len(tokenizer(x["target"]).input_ids)
    x["output_longet_256"] = int(x["output_len"]>256)
    x["output_longet_128"] = int(x["output_len"]>128)
    x["output_longet_64"] = int(x["output_len"]>64)
    return x

sample_size = 10000
data_stats = train_data.select(range(sample_size)).map(map_to_lenght, num_proc=4)

    


# In[14]:


data_stats[0]


# In[15]:


def compute_and_print(x):
    if len(x["input_len"])==sample_size:
        print(
            f"Input mean: {sum(x['input_len'])/sample_size} \n % of input len > 256: {sum(x['input_longer_256'])/sample_size}, \n % of input len > 128: {sum(x['input_longer_128'])/sample_size}, \n % of input len > 64: {sum(x['input_longet_64'])/sample_size}, \n Ouput mean: {sum(x['output_len'])/sample_size},\n% of output len > 256: {sum(x['output_longet_256'])/sample_size}, \n% of output len > 128: {sum(x['output_longet_128'])/sample_size}, \n% of output len > 64: {sum(x['output_longet_64'])/sample_size}")

output = data_stats.map(compute_and_print, batched=True, batch_size=-1)


# In[16]:


train_data


# In[17]:


def convert_to_features(example_batch):
    inputs = tokenizer.batch_encode_plus(example_batch["input"], max_length=64, padding="max_length", truncation=True)
    targets = tokenizer.batch_encode_plus(example_batch["target"], max_length=64, padding="max_length",truncation = True)
    
    encodings = {
        "input_ids":inputs["input_ids"],
        "attention_mask":inputs["attention_mask"],
        "labels":targets["input_ids"],
        "decoder_attention_mask":targets["attention_mask"]
    }
    return encodings

train_data = train_data.map(convert_to_features, batched=True, remove_columns=train_data.column_names)
test_data = test_data.map(convert_to_features, batched=True, remove_columns=test_data.column_names)


# In[ ]:





# In[18]:


columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']

train_data.set_format(type='torch', columns=columns)
test_data.set_format(type='torch', columns=columns)


# In[19]:


len(train_data["input_ids"][12330])


# In[20]:


def sequence_check(list_of_seq):
    len_seq = len(list_of_seq[0])
    for seq in list_of_seq:
        if len(seq)!=len_seq:
            print(len(seq))
            return False
        
    return True


# In[21]:


sequence_check(train_data['input_ids'])


# In[22]:


len(train_data["input_ids"][0])


# In[23]:


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


# In[24]:


training_args = Seq2SeqTrainingArguments(
    output_dir="/home/toibazd/Data/Text2SQL/content/t5-small-finetuned-wikisql",
    per_device_train_batch_size=32,
    num_train_epochs=10,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=500,
    save_strategy="epoch",
    #save_steps=1000,
    #eval_steps=1000,
    overwrite_output_dir=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    push_to_hub=False
    #fp16=True 
)


# In[25]:




# In[32]:


from datasets import load_metric
rouge = load_metric("rouge")


# In[33]:


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


# In[34]:


trainer = Seq2SeqTrainer(
    model = model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset= test_data
)


# In[35]:


trainer.evaluate()


# In[ ]:


trainer.train()


# In[ ]:


trainer.save_model()


# In[ ]:


tokenizer.save_pretrained('/home/toibazd/Data/Text2SQL/content/t5-small-finetuned-wikisql')




# In[ ]:




