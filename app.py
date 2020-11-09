
# Source: https://towardsdatascience.com/rick-and-morty-story-generation-with-gpt2-using-transformers-and-streamlit-in-57-lines-of-code-8f81a8f92692
# Deploy model (--> ./zeh_nlg/model)
# pip install streamlit-nightly==0.69.3.dev20201025
# pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# pip install git+git://github.com/huggingface/transformers@59b5953d89544a66d73


import urllib
import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
model = AutoModelWithLMHead.from_pretrained("./model/trained_model")

model = load_model()


textbox = st.text_area('Start your story:', '', height=200, max_chars=1000)
slider = slider = st.slider('Max text length (in characters)', 50, 1000)
button = st.button('Generate')
if button:
    output_text = model(
            textbox, do_sample=True, max_length=slider, top_k=50, top_p=0.95, num_returned_sequences=1)[0][
            'generated_text'])
