### Text Generation Model based on GPT-2

from __future__ import print_function

import os
from sklearn.model_selection import train_test_split
import io
import tensorflow as tf

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
f = io.open("./data/zeh.txt", mode="r", encoding="utf-8")
data = f.read()
data

train,test = train_test_split(data, test_size=0.2, random_state=42)
train
## tokenizer --> it is important to use the same tokenizer as the tokenizer that was used for the pretrained data

# #split dataset into sentences
#
from nltk.tokenize import sent_tokenize
token_text = sent_tokenize(data, language='german')
print("\nSentence-tokenized copy in a list:")
print(token_text)
print("\nRead the list:")
for s in token_text:
    print(s)

# split into train and test set (is this step really necessary?!)

train, test = train_test_split(token_text,test_size=0.15)
 print(len(token_text))
 print(len(train))
 print(len(test))


from transformers import TextDataset, DataCollatorForLanguageModeling



# MODEL:
from transformers import AutoModelWithLMHead
model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")
model.train()

from transformers import AutoTokenizer, AutoModelWithLMHead


from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")
text_batch = data
encoding = tokenizer(text_batch, return_tensors='pt', padding=False, truncation=False)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
test

test_dataset = TextDataset(tokenizer=tokenizer, file_path=test, block_size=128)
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train, block_size=128)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train
##model
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead


from transformers import AutoModelWithLMHead, Trainer, TrainingArguments

model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train,         # training dataset
    eval_dataset=test,           # evaluation dataset
)

trainer.train()
# generate text
from transformers import pipeline

text_generator = pipeline("text-generation")
print(text_generator("So wie ich das sehe,", max_length=50, do_sample=False))