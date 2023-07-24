from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
import fire
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import mysql.connector
import os

def main(
        model_name: str,
        tokenizer_path: str,
        dataset_path: str,
        output_path: str = "Model_outputs",
        input_max: int = 64,
        output_max: int = 256
):
    model_name = model_name
    tokenizer_path = tokenizer_path
    dataset_path = dataset_path
    input_max = input_max
    output_max = output_max
    parent_directory = output_path

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()


    df = pd.read_csv(dataset_path)
    df = df.sample(frac =1).reset_index(drop=True)
    for index,row in df.iterrows():
        df.loc[index, 'Text'] = "Translate to SQL: " + row['Text']


    test_set_seen = Dataset.from_pandas(df)
    test_set_seen.set_format(type = "torch")

    print(test_set_seen["SQL"][1])
    print(test_set_seen["Text"][1])

    def convert_to_features(example_batch, padding = "max_length",input_max = input_max, output_max = output_max):
        inputs = tokenizer.batch_encode_plus(example_batch["Text"], max_length=input_max, is_split_into_words = False, padding=padding, truncation=True, return_tensors = "pt")
        
        targets = tokenizer.batch_encode_plus(example_batch["SQL"], max_length=output_max, padding = padding,truncation = True)
        if padding == "max_length":
            targets["inputs_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in target] for target in targets["input_ids"]
            ]
        
        inputs["labels"] = targets['input_ids']
        return inputs

    def evaluate_peft_model(sample):
        outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), max_length = 200, top_p=0.9)
        prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        label = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
        label = tokenizer.decode(label, skip_special_tokens=True)
        execution_accuracy(prediction, label)

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
            failed_label_SQL.append(label)

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
    failed_label_SQL = []
    failed_predicted_SQL = []

    with tqdm(total=len(tokenized_dataset), ncols=100, ascii=True) as pbar:
        for sample in tokenized_dataset:
            evaluate_peft_model(sample)
            pbar.set_postfix({'Ac/Ex': accurate_executions})
            pbar.update()



    print("All SQL runs: ", len(all_executions_overall))
    print("Model SQLs that failed: ", len(failed_executions))
    print(f"Execution rate: {len(all_executions_accuracy)/len(all_executions_overall)*100}%")
    print(f"Execution rate: {100 - len(failed_executions)/len(all_executions_overall)*100}%")
    print(f"Execution accuracy: {len(accurate_executions)/len(all_executions_accuracy)*100}%")

    failed_label_sql_df = pd.DataFrame(failed_label_SQL)
    failed_predicted_sql_df = pd.DataFrame(failed_predicted_SQL)
    not_equals = pd.DataFrame({
        'Label':for_checking_label,
        'Prediction': for_checking_prediction
    })

    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    not_equals.to_csv(parent_directory+ "/" + model_name+ "_Not_accurate.csv", index = False)
    failed_label_sql_df.to_csv(parent_directory+"/" + model_name + "_Failed_labels.csv", index = False)
    failed_predicted_sql_df.to_csv(parent_directory+ "/" + model_name +"_Failed_predicted.csv", index = False)



if __name__ == "__main__":
    fire.Fire(main)


