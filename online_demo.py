"""
References:
    https://towardsdatascience.com/spaces-how-to-showcase-your-ml-web-app-demo-in-public-3a701772959
"""

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

tokenizer = T5Tokenizer.from_pretrained('Deep1994/t5-paraphrase-quora')

@st.cache(allow_output_mutation=True)
def load_model():
    model = T5ForConditionalGeneration.from_pretrained('Deep1994/t5-paraphrase-quora')
        
    return model

model = load_model()

st.sidebar.subheader('Select decoding strategy below.')
decoding_strategy = st.sidebar.selectbox("decoding_strategy", ['Top k/p sampling', 'Beam Search'])

st.title('Paraphrase a question in English.')

st.write('This is a fine-tuned t5 model that will paraphrase\
         your English input text into another English output\
         by leveraging a pre-trained [Text-To-Text Transfer Tranformers](https://arxiv.org/abs/1910.10683) model.')


st.subheader('Input Text')
text = st.text_area(' ', height=100)

if text != '':
    set_seed(1234) # for reproducibility
    prefix = 'paraphrase: '
    encoding = tokenizer.encode_plus(prefix + text, padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    if str(decoding_strategy) == 'Top k/p sampling':
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=20,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=5
        )
    elif str(decoding_strategy) == 'Beam Search':
        beam_outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_masks, 
        max_length=20, 
        num_beams=5, 
        no_repeat_ngram_size=2, 
        num_return_sequences=5, 
        early_stopping=True
        )
        
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # if sent.lower() != text.lower() and sent not in final_outputs:
        #     final_outputs.append(sent)
        final_outputs.append(sent)

    st.subheader('Paraphrased Text')
    for i, final_output in enumerate(final_outputs):   
        st.write(final_output + '\n')