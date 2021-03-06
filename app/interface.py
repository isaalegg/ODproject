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


def get_training_path(pathh):
    t_path = os.path.join(pathh, "train", "images")
    v_path = os.path.join(pathh, "val", "images")
    return t_path, v_path


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_revelio_results(image, colors, model):
    clue = model.feature_extractor(image, return_tensors="pt")
    outputs = model.model(**clue)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.
    boxes = rescale_bboxes(
        outputs.pred_boxes[0, keep].cpu(),
        image.size
        )

    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    colors = colors * 100

    # For each bbox
    for p, (xmin, ymin, xmax, ymax), c in zip(probas, boxes.tolist(), colors):
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
        text = f'{model.model.model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    os.remove("result.png")
    plt.savefig("result.png")


def credentials(name, password):
    user = {'name': name, 'pas': password}
    st.session_state.user = user
    print('User credentials are already stored.')


def download_model(train_path, val_path, directory):
    model = ObjectDetectionTrainer(
        'Detr',
        train_path,
        val_path,
        directory,
        start=False
    )
    st.session_state.model = model
    print('the model is already downloaded')


st.title('Revelio Charm')

if 'default_user' not in st.session_state:
    default_name = 'isabel'
    default_password = '123456'
    default_user = {'name': default_name, 'pas': default_password}
    st.session_state.default_user = default_user

if 'user' not in st.session_state:
    st.session_state.user = 'value'
placeholder = st.empty()
with placeholder.container():
    st.subheader('Log in', anchor=None)
    name = st.text_input('User', type="default")
    password = st.text_input('Password', type="password")
    st.button("log in", on_click=credentials, args=(name, password, ))


def presstoinference():
    st.image("result.png")
    print('results are displayed')


if st.session_state.user == st.session_state.default_user:
    placeholder.empty()
    lost_thing = st.radio('tell us what you lost', ['None', 'phone'])
    directory = os.path.dirname(os.path.realpath(__file__))

    if not lost_thing == 'None':
        path = os.path.join(directory, 'dataset', lost_thing)
        st.caption(f'calm down, we will find your {lost_thing}')
        train_path, val_path = get_training_path(path)

        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        label = f'donwload the object detection model'
        if 'model' not in st.session_state:
            st.button(label, on_click=download_model, args=(train_path, val_path, directory, ))
        else:
            file = st.file_uploader(f"what is the last place where you saw it?")
            if file:
                im = Image.open(file)
                get_revelio_results(im, COLORS, st.session_state.model)
                st.image("result.png")
    else:
        st.caption('we can not help you if you do not tell us what you lost, please.')
elif not st.session_state.user == 'value':
    st.write('please, introduce a user and password valid.')
