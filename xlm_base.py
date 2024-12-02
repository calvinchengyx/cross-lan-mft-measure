#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
from imblearn.under_sampling import RandomUnderSampler

import argparse # Step 1: Import the argparse module


####### PREPARE PATHs VARAIBLES #########
# # the only iteratable subset number, there are 15 subsets in total
base_dir= "/xml-t-base-english"
# experiment_dir = "3_1_"
# subset_dir = "english"
model_trained_dir = "_best_model"

# Step 2: Define the argument
parser = argparse.ArgumentParser(description='Specify the foundation to be trained.')
parser.add_argument('--foundation', type=str, required=True, help='The foundation to be trained')  
parser.add_argument('--gpu', type=str, required=True, help='specify the GPU to be used')  

args = parser.parse_args() # Step 3: Parse the arguments

# Step 4: Use the parsed value
foundation = args.foundation 
gpu = args.gpu
# train_subset_dir = "train_subsets/"

# change all the operations to the base directory
os.chdir(base_dir)

#### load parameters #####
LR = 2e-5 # learning rate, this is the default value, one of the hyperparameters fine-tuned in Mformer
EPOCHS = 3 # from Paul's paper, the optional first phase of fine-tuning on English data was for three epochs.
BATCH_SIZE = 16 # from Paul's paper, the training batch size is set to 16, same with Mformer
MODEL = "cardiffnlp/twitter-xlm-roberta-base" # use this to finetune the language model
#MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment" # use this to finetune the sentiment classifier
MAX_TRAINING_EXAMPLES = -1 # set this to -1 if you want to use the whole training set 

##### load annotated English data where dropping non-moral category, remember to reset the index, from the merged_corp #####
dataset = pd.read_csv('/e_tools_lm/data/mf_corpora_merged.csv', index_col=0)
# double check and make sure all data text are strings, no nan values
mask = dataset['sentence'].apply(lambda x: isinstance(x, str)) 
dataset = dataset[mask]
# reset the index
dataset.reset_index(drop=True, inplace=True) 
dataset = dataset[dataset[f'{foundation}_label'] != -1] # -1 is the label for no corresponding label in the foundation
dataset = dataset[['sentence', f'{foundation}_label']]
dataset = dataset.rename(columns={'sentence': 'text', f'{foundation}_label': 'label'})


df_train_imbalance, df_temp = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['label']) # 80% for training 

# undersample the training data
def undersample_training_data(df_train, label_column='label', random_state=42):
    """
    Undersample the majority class in the training data.

    Parameters:
    df_train (pd.DataFrame): The training dataframe containing features and labels.
    label_column (str): The name of the label column. Default is 'label'.
    random_state (int): The random state for reproducibility. Default is 42.

    """
    # Separate features and labels
    X_train = df_train.drop(columns=[label_column])
    y_train = df_train[label_column]

    # Initialize the undersampler
    undersampler = RandomUnderSampler(random_state=random_state)

    # Perform undersampling
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

    # Combine the resampled features and labels back into a dataframe
    df_train = pd.concat([X_train_resampled, y_train_resampled], axis=1)

    return df_train

# undersample the training data, no need for validation and test data
df_train = undersample_training_data(df_train_imbalance)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['label']) # 10% for validation and 10% for in-sample testing 


##### load the data #####
# prepare for the data, skip the downloading part if you already prepared the data
dataset_dict = {}
dataframes = {'train': df_train, 'val': df_val, 'test': df_test}

for i in ['train', 'val', 'test']:
    dataset_dict[i] = {}
    df = dataframes[i]
    dataset_dict[i]['text'] = df['text'].tolist()
    dataset_dict[i]['label'] = df['label'].tolist()

#### Checkpoint 1. the dictionary #####, dump the diciontary to files and check it 

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, max_length=512, padding='max_length')
val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, max_length=512, padding='max_length')
test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, max_length=512, padding='max_length')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # self.encoding is the dictionary result from tokenizer() function, with two keys: input_ids and attention_mask, each has 512 dimension values
        # this iteration is to convert each key-value pair to a tensor
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, dataset_dict['train']['label'])
val_dataset = MyDataset(val_encodings, dataset_dict['val']['label'])
test_dataset = MyDataset(test_encodings, dataset_dict['test']['label'])


# Optionally, set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = gpu # Use GPU 2

# set the training Arguments:
training_args = TrainingArguments(
    output_dir='/xml-t-base-english/3_1_results',                   # output directory
    num_train_epochs=EPOCHS,                  # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
    warmup_steps=100,                         # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                        # strength of weight decay
    logging_dir='/xml-t-base-english/3_1_logs',                     # directory for storing logs
    logging_steps=10,                         # when to print log
    evaluation_strategy="steps",
    save_strategy="steps",     
    load_best_model_at_end=True              # load or not best model at the end
    # max_steps=10                    # for the time estimation purpose, comment it out  after testing
)


# Model Initialization:
num_labels = len(set(dataset_dict["train"]["label"]))
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)


trainer = Trainer(
    model=model,                              # the instantiated 🤗 Transformers model to be trained
    args=training_args,                       # training arguments, defined above
    train_dataset=train_dataset,              # training dataset
    eval_dataset=val_dataset                  # evaluation dataset
)

# start training the model, with the current setting, it may take 240min (4 hours) to finish the training
# 44k training documents, one A100 GPU, RoBERTa-base model, batch size 16, number of epochs 3
trainer.train()

trainer.save_model(f"/e_tools_lm/finetune_xml/en_base_{foundation}") # save best model 
print("The model has been successfully saved")


# # E3.1.3 evaluate the model with test dataset (noted, not benchmark dataset yet)
test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_raw, axis=1)
report = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)

print(f"The {foundation} base model performance on in-sample test dataset:")
report_df = pd.DataFrame(report).transpose()
print(report_df)
