import streamlit as st
from pathlib import Path
import os
import pandas as pd
import numpy as np


st.title('Revelio Charm')

# options should be possible to extract to dataset list
losed_thing = st.radio('tell me what you lose', ['None', 'phone', 'keys'])
directory = os.path.dirname(os.path.realpath(__file__))
path = Path(directory).parents[1]
selected_data = os.path.join(path, 'core', 'dataset', losed_thing)
if losed_thing == 'phone':
    st.write('we have the solution for your problem! Give us a moment.')

magic = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
image = st.file_uploader(f"Where you think what you lose your {losed_thing}")
clue = magic.extractor(image, return_tensors="pt")


def start():
    label = 'click me'
    if st.button(label):
        model = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
