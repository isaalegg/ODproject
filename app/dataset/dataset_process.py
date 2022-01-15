from transformers import DetrFeatureExtractor
from torch.utils.data import DataLoader

import numpy as np
import os
import torchvision
from PIL import Image, ImageDraw


class ODDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder):
        ann_file = os.path.join(img_folder, "coco_instances_more-imgs.json")
        super(ODDataset, self).__init__(img_folder, ann_file)
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(ODDataset, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    def dataloader(self, dataset):
        dataloader = DataLoader(dataset, collate_fn=self.collate_fn, batch_size=2)
        return dataloader



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

