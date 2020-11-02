# Juli Zeh Lyrics Generator
## Natural Language Generation Neural Net
### *pretrained GPT-2 for text generation in German*
------------

## Installation:
For installation see: [huggingface](https://huggingface.co/transformers/installation.html)
Transformers can either be based on  Tensorflow, Pytorch or both. 
The present model is based on Tensorflow.

Model pre-trained on German text and fine-tuned on novel "Spieltrieb" by Juli Zeh. Data contains only parts of the novel with randomized chapters to prevent copyright violations.  

## Content: 

1. Installation inside venv: 1. TF and/or 2. Pytorch and 3. transformers
1. install dependencies
2. load the dataset
1. prepare the dataset and build a TextDataset
1. load the pre-trained German-GPT-2 model and tokenizer
1.initialize Trainer with Training Arguments
1. train and save the model
1. test the model
