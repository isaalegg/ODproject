import os
import sys
sys.path.append(os.getcwd())
from app.magic.trainer import ObjectDetectionTrainer
from PIL import Image
import streamlit as st
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


st.title('Revelio Charm')


def get_training_path(pathh):
    t_path = os.path.join(pathh, "train", "images")
    v_path = os.path.join(pathh, "val", "images")
    return t_path, v_path


def get_revelio_results(image, colors, model):
    clue = model.feature_extractor(image, return_tensors="pt")
    outputs = model.model(**clue)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    target_sizes = torch.tensor(im.size[::-1]).unsqueeze(0)
    postprocessed_outputs = model.feature_extractor.post_process(outputs, target_sizes)
    boxes = postprocessed_outputs[0]['boxes'][keep]

    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    colors = colors * 100

    # For each bbox
    for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], boxes.tolist(), colors):
        # Draw the bbox as a rectangle
        ax.add_patch(plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            color=c,
            linewidth=3
        ))

        # Get the highest probability
        cl = p.argmax()

        # Draw the label
        text = f'{model.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.savefig('result.png')


losed_thing = st.radio('tell us what you lost', ['None', 'phone'])
directory = os.path.dirname(os.path.realpath(__file__))

if not losed_thing == 'None':
    path = os.path.join(directory, 'dataset', losed_thing)
    st.write(f'calm down, we will find your {losed_thing}')
if losed_thing == 'None':
    st.write('we can not help you if you do not tell us what you lost, please.')

train_path, val_path = get_training_path(path)

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

label = f'find your {losed_thing}'
if 'key' not in st.session_state:
    click_there = st.button(label)
    if click_there:
        model_class = ObjectDetectionTrainer('Detr', train_path, val_path, directory, start=False)
        st.session_state.key = model_class.model.model
else:
    file = st.file_uploader(f"what is the last place where you saw it?")
    if file:
        im = Image.open(file)
        magic = st.session_state.key
        get_revelio_results(im, COLORS, magic)
        st.image('result.png')
