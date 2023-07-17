#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextStreamer

hf_peft_repo = "LoRA_adapted_T5"
peft_config = PeftConfig.from_pretrained(hf_peft_repo)
model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
streamer = TextStreamer(tokenizer)

# Load the finetuned Lora PEFT model
model = PeftModel.from_pretrained(model, hf_peft_repo)
model.eval()


# In[2]:


from datasets import Dataset
import pandas as pd

df = pd.read_csv('Training_set_IMDB/testing_set_no_target.csv')
df = df.sample(frac =1).reset_index(drop=True)
for index,row in df.iterrows():
    df.loc[index, 'Text'] = "Translate to SQL: " + row['Text']


df2 = pd.read_csv('Training_set_IMDB/testing_set_unseen_no_target.csv')
df2 = df2.sample(frac =1).reset_index(drop=True)
for index,row in df2.iterrows():
    df2.loc[index, 'Text'] = "Translate to SQL: " + row['Text']

test_set_seen = Dataset.from_pandas(df)
test_set_unseen = Dataset.from_pandas(df2)


# In[3]:


test_set_seen.set_format(type = "torch")
test_set_unseen.set_format(type = "torch")

test_set_seen["Text"][1]


# In[4]:


def map_to_lenght(x):
    x["input_len"] = len(tokenizer(x["Text"]).input_ids)
    x["input_longer_256"] = int(x["input_len"]>256)
    x["input_longer_128"] = int(x["input_len"]>128)
    x["input_longet_64"] = int(x["input_len"]>64)
    x["output_len"] = len(tokenizer(x["SQL"]).input_ids)
    x["output_longet_256"] = int(x["output_len"]>256)
    x["output_longet_128"] = int(x["output_len"]>128)
    x["output_longet_64"] = int(x["output_len"]>64)
    return x

sample_size = 2000
data_stats = test_set_seen.select(range(sample_size)).map(map_to_lenght, num_proc=4)
data_stats_2 = test_set_unseen.select(range(sample_size)).map(map_to_lenght, num_proc=4)


# In[5]:


def compute_and_print(x):
    if len(x["input_len"])==sample_size:
        print(
            f"Input mean: {sum(x['input_len'])/sample_size} \n % of input len > 256: {sum(x['input_longer_256'])/sample_size}, \n % of input len > 128: {sum(x['input_longer_128'])/sample_size}, \n % of input len > 64: {sum(x['input_longet_64'])/sample_size}, \n Ouput mean: {sum(x['output_len'])/sample_size},\n% of output len > 256: {sum(x['output_longet_256'])/sample_size}, \n% of output len > 128: {sum(x['output_longet_128'])/sample_size}, \n% of output len > 64: {sum(x['output_longet_64'])/sample_size}")

output = data_stats.map(compute_and_print, batched=True, batch_size=-1)


# In[6]:


test_set_seen['Text'][0]


# In[7]:


def convert_to_features(example_batch, padding = "max_length",input_max = 100, output_max = 170):
    inputs = tokenizer.batch_encode_plus(example_batch["Text"], max_length=input_max, is_split_into_words = False, padding='max_length', truncation=True, return_tensors = "pt")
    
    targets = tokenizer.batch_encode_plus(example_batch["SQL"], max_length=output_max, padding = "max_length",truncation = True)
    if padding == "max_length":
        targets["inputs_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in target] for target in targets["input_ids"]
        ]
    
    inputs["labels"] = targets['input_ids']
    return inputs

def evaluate_peft_model(sample):
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), max_new_tokens = 200, top_p=0.9)
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    label = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    label = tokenizer.decode(label, skip_special_tokens=True)
    _ = execution_accuracy(prediction, label)
    
    return prediction, label

def execution_accuracy(prediction, label):
    try:
        
        cursor.execute(label)
        result_label = cursor.fetchall()
        all_executions_overall.append(1)
        try:
            cursor.execute(prediction)
            result_pred = cursor.fetchall()
            all_executions_accuracy.append(1)
            if len(result_label)>10:
                if len(result_label) == len(result_pred):
                    accurate_executions.append(1) 
            elif result_label == result_pred:
                accurate_executions.append(1)
            else:
                for_checking_label.append(label)
                for_checking_prediction.append(prediction)
                
        except:
            failed_executions.append(1)
            failed_predicted_SQL.append(prediction)
                
    except:
        failed_original_SQL.append(label)
    return None



# In[8]:


import evaluate
import numpy as np
from tqdm import tqdm
import mysql.connector



connection = mysql.connector.connect(
    host="relational.fit.cvut.cz",
    user="guest",
    password="relational",
    database="imdb_ijs"
)
cursor = connection.cursor()



print("mapping both datasets")
tokenized_dataset = test_set_seen.map(convert_to_features, batched=True, num_proc=4)
tokenized_dataset_2 = test_set_unseen.map(convert_to_features, batched=True, num_proc=4)
print("mapped both dataset")
print("Documents we have: tokenizer_dataset for seen and tokenized_dataset_2 for unseen data")


print("\n\n Running executions for seen dataset")
all_executions_overall = []
failed_executions = []
all_executions_accuracy = []
accurate_executions = []
for_checking_label = []
for_checking_prediction = []
failed_original_SQL = []
failed_predicted_SQL = []



for sample in tokenized_dataset:
    p,l = evaluate_peft_model(sample)
    print("all_executions_overall: ", len(all_executions_overall))
    print("all_executions_accuracy: ",len(all_executions_accuracy))
    print("accurate_executions: ", len(accurate_executions))
    print("failed_executions: ", len(failed_executions))


print("All SQL runs:", len(all_executions_overall))
print("Model SQLs that failed: ", len(failed_executions))
print(f"Execution rate: {len(all_executions_accuracy)/len(all_executions_overall)*100}%")
print(f"Execution rate: {100 - len(failed_executions)/len(all_executions_overall)*100}%")
print(f"Execution accuracy: {len(accurate_executions)/len(all_executions_accuracy)*100}%")

failed_original_sql_df = pd.DataFrame(failed_original_SQL)
failed_predicted_sql_df = pd.DataFrame(failed_predicted_SQL)
not_equals = pd.DataFrame({
    'Label':for_checking_label,
    'Prediction': for_checking_prediction
})


not_equals.to_csv("/home/toibazd/Data/Text2SQL/Training_set_IMDB/Not_equals_lora.csv", index = False)
failed_original_sql_df.to_csv("/home/toibazd/Data/Text2SQL/Training_set_IMDB/Failed_originals_lora.csv", index = False)
failed_predicted_sql_df.to_csv("/home/toibazd/Data/Text2SQL/Training_set_IMDB/Failed_predicted_lora.csv", index = False)


# In[ ]:


print("\n \n Running evaluation on unseen data")
all_executions_overall = []
failed_executions = []
all_executions_accuracy = []
accurate_executions = []
for_checking_label = []
for_checking_prediction = []
failed_original_SQL = []
failed_predicted_SQL = []

for sample in tokenized_dataset_2:
    p,l = evaluate_peft_model(sample)

print("All SQL runs:", len(all_executions_overall))
print("Model SQLs that failed: ", len(failed_executions))
print(f"Execution rate: {len(all_executions_accuracy)/len(all_executions_overall)*100}%")
print(f"Execution rate: {100 - len(failed_executions)/len(all_executions_overall)*100}%")
print(f"Execution accuracy: {len(accurate_executions)/len(all_executions_accuracy)*100}%")


# In[ ]:





# In[ ]:





# In[ ]:




