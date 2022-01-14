import streamlit as st
import convert as via2coco
from pathlib import Path
import sys
import os
sys.path.append(os.getcwd())
from app.magic.model import Model
from app.dataset.dataset_process import ODDataset

st.title('Revelio Charm')

# me esta agarrando el val, arregla eso
losed_thing = st.radio('tell me what you lose', ['phone','Balloon' 'None'])
directory = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(directory, 'dataset', losed_thing)
if losed_thing == 'phone':
    selected_data = os.path.join(path, "train" and 'val')
    st.write('we have the solution for your problem! Give us a moment.')
elif losed_thing == 'balloon':
    st.write('we have the solution for your problem! Give us a moment.')

def convert_format(selected_data):
    first_class_index = 0

    for keyword in ['train', 'val']:
        input_dir = selected_data + '/' + keyword + '/'
        input_json = input_dir + 'via_region_data.json'
        categories = ['balloon']
        super_categories = ['N/A']
        output_json = input_dir + 'custom_' + keyword + '.json'

        coco_dict = via2coco.convert(
            imgdir=input_dir,
            annpath=input_json,
            categories=categories,
            super_categories=super_categories,
            output_file_name=output_json,
            first_class_index=first_class_index,
        )

def id2label_to_model(data):
    dataset = ODDataset(data)
    cats = dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}
    return id2label

labels = id2label_to_model(selected_data)
magic = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, labelid=labels)
image = st.file_uploader(f"Where you think what you lose your {losed_thing}")
clue = magic.extractor(image, return_tensors="pt")

def start():
    label = 'click me'
    if st.button(label):
        model = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
