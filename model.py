### Text Generation Model based on GPT-2

from __future__ import print_function

import csv

import numpy as np
import io
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import time

############################ 28/10/2020

# where am I?
os.getcwd()

# conda activate zeh2
# Installation inside venv: 1. TF and/or 2. Pytorch and 3. transformers
# 1. install dependencies
# 2. load the dataset
# prepare the dataset and build a TextDataset
# load the pr e -trained GP T -2 model and tokenizer
# initialize Trainer with TrainingArguments
# train and save the model
# test the model





## load data:
path = open('./data/zeh.csv', encoding='utf-8')
data = path.readlines()
data

import re
import csv
from sklearn.model_selection import train_test_split

import csv

import pandas as pd
import io
f = io.open("./data/zeh.txt", mode="r", encoding="utf-8")
data = f.read()
data

from nltk.tokenize import sent_tokenize
token_text = sent_tokenize(data, language='german')
print("\nSentence-tokenized copy in a list:")
print(token_text)
print("\nRead the list:")
for s in token_text:
    print(s)

train, test = train_test_split(token_text,test_size=0.15)
print(len(token_text))
print(len(train))
print(len(test))
test


def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)
###
from transformers import TextDataset, DataCollatorForLanguageModeling



# MODEL:

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")

model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")

# generate text
from transformers import pipeline

text_generator = pipeline("text-generation")
print(text_generator("So wie ich das sehe,", max_length=50, do_sample=False))