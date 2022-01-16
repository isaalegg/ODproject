import streamlit as st
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from PIL import Image
from datetime import datetime
import sys
import os
sys.path.append(os.getcwd())
from app.magic.model import Detr
from app.dataset.collator import ObjectDetectionCollator
sys.path.append(os.getcwd())

st.title('Revelio Charm')

# me esta agarrando el val, arregla eso
losed_thing = st.radio('tell me what you lose', ['None', 'phone'])
directory = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(directory, 'dataset', losed_thing)
if not losed_thing == 'None':
    train_data = os.path.join(path, "train")
    val_data = os.path.join(path, "val")
    st.write('we have the solution for your problem! Give us a moment.')
elif losed_thing == 'None':
    st.write('select any option, please.')


def id2label_to_model(data):
    dataset = Detr(data)
    cats = dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}
    return id2label

def get_training_data(path):
    train_data = os.path.join(path, "train")
    val_data = os.path.join(path, "val")
    train_dataset = Detr(train_data)
    val_dataset = Detr(val_data)
    return train_dataset, val_dataset


labels = id2label_to_model(train_data)
train_dataloader, val_dataloader = get_training_data(path)
magic = Detr(
    lr=1e-4,
    lr_backbone=1e-5,
    weight_decay=1e-4,
    labelid=labels,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader
)


file = st.file_uploader(f"Where you think what you lose your {losed_thing}")
if file:
    im = Image.open(file)
    clue = magic.feature_extractor(im, return_tensors="pt")
    extractor_outputs = magic.model(**clue)
    probas = extractor_outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def postprocess_outputs(outputs, img, keep):
    target_sizes = torch.tensor(img.size[::-1]).unsqueeze(0)
    postprocessed_outputs = magic.feature_extractor.post_process(outputs, target_sizes)
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

    colors = COLORS * 100

    # For each bbox
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
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
    plt.savefig('result.png' + datetime.today().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg")


results = get_revelio_results(im, COLORS, probas[keep], pp_output, magic)
st.image('result.png')

collator = ObjectDetectionCollator(magic.feature_extractor)
train_dataset, val_dataset = get_training_data(path)

train_dataloader = DataLoader(
        train_dataset,
        collate_fn  = collator,
        batch_size  = 2,
        shuffle     = True,
        num_workers = int(os.cpu_count() * 0.75),
    )

val_dataloader = DataLoader(
        val_dataset,
        collate_fn  = collator,
        batch_size  = 2,
        num_workers = int(os.cpu_count() * 0.75),
    )

label = 'click me'
if st.button(label):
    model_base = Detr(
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        labelid=labels,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
)
    trainer = Trainer()
    trainer.fit(model_base)
    model_base.model.save_pretrained(os.path.join(path, "new_model"))