import streamlit as st
import pandas as pd
import numpy as np

st.title('Revelio Charm')

label = 'click me'
# options should be possible to extract to dataset list
losed_thing = st.radio('tell me what you lose', ['None', 'phone', 'keys'])
if st.button(label):
    st.title('woa, its magic')
if losed_thing == 'phone':
    st.write('I will return the phone dataset to train')

st.file_uploader(f"Where you think what you lose your {losed_thing}")