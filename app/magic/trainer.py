import os
from datetime import datetime

import torch
import torchmetrics
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from app.magic.model import Detr
from app.dataset.dataset_process import CocoDetection
from app.dataset.collator import ObjectDetectionCollator


from transformers import DetrFeatureExtractor


class ObjectDetectionTrainer:

    def __init__(
            self,
            model_name: str,
            train_path: str,
            val_path: str,
            output_dir: str,
            lr=1e-4,
            lr_backbone=1e-5,
            batch_size=4,
            max_epochs=1,
            shuffle=True,
            augmentation=False,
            weight_decay=1e-4,
            max_steps=1,
            nbr_gpus=0,
            model_path="facebook/detr-resnet-50",
            start=bool,
    ):

        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.output_dir = output_dir
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.nbr_gpus = nbr_gpus
        self.model_path = model_path
        self.start = start

        # Processing device (CPU / GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Setup the metric
        self.metric = torchmetrics.Accuracy()

        # Load feature extractor
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_path)

        # Get the classifier collator
        self.collator = ObjectDetectionCollator(self.feature_extractor)

        # Get the model output path
        self.output_path = self.__getOutputPath()
        self.logs_path = self.output_path

        # Open the logs file
        self.__openLogs()

        # Split and convert to dataloaders
        self.train, self.val, = self.__splitDatasets()

        # Get labels and build the id2label

        # print("*"*100)
        categories = self.train_dataset.coco.dataset['categories']
        self.id2label = {}
        self.label2id = {}
        for category in categories:
            self.id2label[category['id']] = category['name']
            self.label2id[category['name']] = category['id']

        print(self.id2label)
        print(self.label2id)

        self.model = Detr(
            lr=self.lr,
            lr_backbone=self.lr_backbone,
            weight_decay=self.weight_decay,
            id2label=self.id2label,
            label2id=self.label2id,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            model_path=self.model_path,
        )


        self.trainer = Trainer(
            gpus=self.nbr_gpus,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            gradient_clip_val=0.1
        )
        print("Trainer builded!")

        # Fine-tuning
        if self.start == True:
            print("Start Training!")

            self.trainer.fit(self.model)

            # Save for huggingface
            self.model.basemodel.save_pretrained(self.output_path)
            print("Model saved at: \033[93m" + self.output_path + "\033[0m")

            # Close the logs file
            self.logs_file.close()



    def __openLogs(self):

        # Open the logs file
        self.logs_file = open(self.logs_path + "/logs.txt", "a")


    def __getOutputPath(self):

        path = os.path.join(
            self.output_dir,
            self.model_name.upper() + "/" + str(self.max_epochs) + "_" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        )

        # Create the full path if doesn't exist yet
        if not os.path.isdir(path):
            os.makedirs(path)

        return path

    def __splitDatasets(self):

        print("Load Datasets...")

        # Train Dataset in the COCO format
        self.train_dataset = CocoDetection(
            img_folder=self.train_path,
            feature_extractor=self.feature_extractor
        )

        # Val Dataset in the COCO format
        self.val_dataset = CocoDetection(
            img_folder=self.val_path,
            feature_extractor=self.feature_extractor
        )

        print(self.train_dataset)
        print(self.val_dataset)

        workers = int(os.cpu_count() * 0.75)

        # Train Dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=workers,
        )

        # Validation Dataloader
        self.val_dataloader = DataLoader(
            self.val_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            num_workers=workers,
        )


        return self.train_dataloader, self.val_dataloader
