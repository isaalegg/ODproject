import os
import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder):
        # Get path image annotations
        ann_file = os.path.join(img_folder, "coco_instances_more-imgs.json")

        # Default constructor
        super(CocoDetection, self).__init__(img_folder, ann_file)

        # Load the feature extractor

    def __getitem__(self, idx):
        # Read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # Preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

