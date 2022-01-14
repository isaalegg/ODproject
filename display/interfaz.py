import streamlit as st
from pathlib import Path
import sys
import os
sys.path.append(os.getcwd())
from display.magic.model import Model
from display.dataset.dataset_process import ODDataset

st.title('Revelio Charm')

# debuggear ese path porque cambiaron los componentes
losed_thing = st.radio('tell me what you lose', ['None', 'phone', 'balloon'])
directory = os.path.dirname(os.path.realpath(__file__))
path = Path(directory).parents[1]
selected_data = os.path.join(path, 'dataset', losed_thing)
if losed_thing == 'phone':
    st.write('we have the solution for your problem! Give us a moment.')
elif losed_thing == 'balloon':
    st.write('we have the solution for your problem! Give us a moment.')


def id2label_to_model():
    dataset = ODDataset(selected_data)
    cats = dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}
    return id2label

labels = id2label_to_model()
magic = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, labelid=labels)
image = st.file_uploader(f"Where you think what you lose your {losed_thing}")
clue = magic.extractor(image, return_tensors="pt")

def start():
    label = 'click me'
    if st.button(label):
        model = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
