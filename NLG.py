#from transformers import AutoModelWithLMHead, AutoTokenizer

#model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased", return_dict=True)
#tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")
model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")


# Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
PADDING_TEXT = data[1:500]
print(PADDING_TEXT)
prompt = "Ada liebte es, wenn"
inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")

max_length = len(PADDING_TEXT)

prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
generated