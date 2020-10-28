### Text Generation Model based on GPT-2

# 1. install dependencies
# 2. load the dataset
#prepare the dataset and build a TextDataset
#load the pr e -trained GP T -2 model and tokenizer
#initialize Trainer with TrainingArguments
#train and save the model
#test the model



# conda activate zeh2
# Installation inside venv: 1. TF and/or 2. Pytorch and 3. transformers


from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")

model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")
