import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrFeatureExtractor
import torch
from pytorch_lightning import Trainer
from core.app.dataset.dataset_process import CocoDetection

class Model(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.id2tolabel = CocoDetection.id2label_to_model()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=len(self.id2label),
                                                            ignore_mismatched_sizes=True)
        #id2label is a feature defined in dataset class, calm down
        # The learning rate (lr) controls how quickly the model is adapted to the problem.
        # Smaller learning rates require more training epochs given the smaller changes made to the weights each update,
        # whereas larger learning rates result in rapid changes and require fewer training epochs.

        #the "backbone" refers to the feature extracting network which is used within the DeepLab architecture.
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    def preprocessing_app_input(self, img):
        encoding_img = self.extractor(img, return_tensors="pt")
        return encoding_img

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader
    #dont worry, this and below features are defined in dataset class.

    def val_dataloader(self):
        return val_dataloader


    def new_model(self):
        trainer = Trainer(gpus=1, max_steps=300, gradient_clip_val=0.1)
        trainer.fit(model)
