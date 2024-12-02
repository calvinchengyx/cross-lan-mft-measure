from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
# for model training using hggingface datasets, for the loading, processing, and sharing of datasets for machine learning and data science tasks
from datasets import load_dataset, Dataset
# A deep learning library used for tensor computations and building neural networks.
import torch
# for the evaluation of the model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd

# define the argument
import argparse

##### 1: Define the argument #####
parser = argparse.ArgumentParser(description= 'Specify necessary variables to infer')
parser.add_argument('--foundation', type=str, required=True, help='specify the foundation infer')  
parser.add_argument('--gpu', type=str, default='2', required=True, help='specify the GPU to be used')
parser.add_argument('--modelpath', type=str, required=True, help='specify the model to be used')
parser.add_argument('--output_col_name', type=str, required=True, help='specify the inference column name prefix in the result csv, e.g. _base, _ch')

args = parser.parse_args() # Step 3: Parse the arguments

# Use the parsed value
foundation = args.foundation 
gpu = args.gpu
modelpath = args.modelpath
output_col_name = args.output_col_name

##### 2: Load the model #####
model = AutoModelForSequenceClassification.from_pretrained(modelpath)
 # keep using the same xlm-tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")

# set the GPU device, GPU 2 is the first A100 one
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

######### load the test data ############
df_test = pd.read_csv("/data/scro4316/thesis/paper3/benchmarkset_map_0917.csv")

test_encodings = tokenizer(df_test['text'].tolist(), truncation=True, max_length=512, padding='max_length')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = MyDataset(test_encodings, df_test[f'{foundation}_label'].tolist())

# load the model 
trainer = Trainer(model=model) # the instantiated 🤗 Transformers model to be trainer

# run the inference
test_preds_raw, test_labels , _  = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_raw, axis=1)

# write the result to a existing dataframe (copied from /data/scro4316/thesis/paper3/benchmarkset_map_0917.csv)
df = pd.read_csv("/data/scro4316/thesis/paper3/e_tools_lm/finetune_xml/raw_result_xml.csv")
df[f'pred_{foundation}_{output_col_name}'] = test_preds
# save the result and replace the existing one
df.to_csv(f"/data/scro4316/thesis/paper3/e_tools_lm/finetune_xml/raw_result_xml.csv", index=False)
print("Inference done and saved to raw_result_xml.csv")