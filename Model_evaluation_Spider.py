#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import pandas as pd


# In[2]:


model_name = "T5-Finetuned-with-IMDB-Spider"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device


# In[3]:


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()


# In[6]:


from datasets import Dataset

df = pd.read_csv('Training_set_IMDB/testing_set_no_target.csv')
df = df.sample(frac =1).reset_index(drop=True)
for index,row in df.iterrows():
    df.loc[index, 'Text'] = "Translate to SQL: " + row['Text']


test_set_seen = Dataset.from_pandas(df)


# In[10]:


test_set_seen.set_format(type = "torch")

print(test_set_seen["SQL"][1])
print(test_set_seen["Text"][1])


# In[13]:


def convert_to_features(example_batch, padding = "max_length",input_max = 64, output_max = 256):
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



# In[15]:


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

print("mapped both dataset")
print("Document we have: tokenized_dataset for seen data")


print("\n\n Running executions for seen dataset")
all_executions_overall = []
failed_executions = []
all_executions_accuracy = []
accurate_executions = []
for_checking_label = []
for_checking_prediction = []
failed_original_SQL = []
failed_predicted_SQL = []



for sample in tqdm(tokenized_dataset):
    p,l = evaluate_peft_model(sample)



print("All SQL runs: ", len(all_executions_overall))
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

not_equals.to_csv("/home/toibazd/Data/Text2SQL/Training_set_IMDB/Not_equals_FT.csv", index = False)
failed_original_sql_df.to_csv("/home/toibazd/Data/Text2SQL/Training_set_IMDB/Failed_originals_Spider.csv", index = False)
failed_predicted_sql_df.to_csv("/home/toibazd/Data/Text2SQL/Training_set_IMDB/Failed_predicted_Spider.csv", index = False)


# In[ ]:




