# Information: https://huggingface.co/transformers/training.html

# import tensorflow as tf
# import transformers
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
    block_size=128
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
    output_dir="halde/old_models/args",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
p    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
)

# now TRAIN!
trainer.train()
trainer.save_model("./model/trained_model")
tokenizer.save_pretrained("./model/toke")


german_model = AutoModelWithLMHead.from_pretrained("./model/trained_model")
#doesnt work and may not be need - didnt make changes to tokenizer anyways, using "tokenizer"
#german_tokenizer = AutoTokenizer.from_pretrained("./model/toke")


prompt = "Ada liebte ihre Katze"
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
max_length = 150
prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))

# for provided options see huggingface blog https://huggingface.co/blog/how-to-generate
# top_k => In Top-K sampling, the K most likely next words are filtered and the probability mass is redistributed among only those K next words.
# top_p => Having set p=0.95, Top-p sampling picks the minimum number of words to exceed together p=.95% of the probability mass
outputs = german_model.generate(inputs, max_length = max_length, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=2)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
lyric = generated.replace('\n', ' ')
print(lyric)

# from transformers import pipeline
#
# text_generator = pipeline("text-generation",model=german_model,tokenizer=tokenizer)
# print(text_generator("Ada liebte ihre Katze und", max_length=50, do_sample=False))
