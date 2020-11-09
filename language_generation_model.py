# Information: https://huggingface.co/transformers/training.html

import tensorflow as tf
import transformers
from transformers import LineByLineTextDataset

# import tokenizer and model
from transformers import AutoModelWithLMHead, AutoTokenizer

# load tokenizer and model
# ! for pretraining and fine-tuning the same tokenizer must be used
tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")
model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")
#model.num_parameters()
# --> 125 million parameters!

#load dataset in format that model can read
train_data = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/zeh.txt",
    block_size=128,
)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

# make dir if not done before:
#os.chdir("path/to/working/directory")
#os.mkdir("./model/trained_model")
#os.mkdir("./model/args")

# GPT-2 does not work with Tensorflow Trainer or TrainingArguments!

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model/args",
    overwrite_output_dir=True,
    num_train_epochs=200,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
)

# now TRAIN!
trainer.train()

trainer.save_model("./model/trained_model")
tokenizer.save_pretrained("./model/toke")

model = AutoModelWithLMHead.from_pretrained("./model/trained_model")


prompt = "Ada liebte ihre Katze"
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
max_length = 150
prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length = max_length, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
generated
s