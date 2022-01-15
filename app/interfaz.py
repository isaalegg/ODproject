import streamlit as st
import torch
import matplotlib.pyplot as plt
import convert as via2coco
from PIL import Image
from pathlib import Path
import sys
import os
sys.path.append(os.getcwd())
from app.magic.model import Model
from app.dataset.dataset_process import ODDataset

st.title('Revelio Charm')

# me esta agarrando el val, arregla eso
losed_thing = st.radio('tell me what you lose', ['phone','balloon', 'None'])
directory = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(directory, 'dataset', losed_thing)
if not losed_thing == 'None':
    selected_data = os.path.join(path, "train")
    st.write('we have the solution for your problem! Give us a moment.')
elif losed_thing == 'None':
    st.write('select any option, please.')


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

file = st.file_uploader(f"Where you think what you lose your {losed_thing}")
im = Image.open(file)
clue = magic.extractor(im, return_tensors="pt")
extractor_outputs = magic(**clue)
probas = extractor_outputs.logits.softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def postprocess_outputs(outputs, img, keep):
    target_sizes = torch.tensor(img.size[::-1]).unsqueeze(0)
    postprocessed_outputs = magic.extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    return bboxes_scaled



pp_output = postprocess_outputs(
    extractor_outputs,
    im,
    keep
)

def get_revelio_results(im, COLORS, prob, boxes, model):
    plt.figure(figsize=(16, 10))
    plt.imshow(im)
    ax = plt.gca()
    paint = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), paint):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{model.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('result.png', bbox_inches='tight')

results = get_revelio_results(im, COLORS, probas[keep], pp_output, magic)
st.image('result.png')


def start():
    label = 'click me'
    if st.button(label):
        model = Model(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
