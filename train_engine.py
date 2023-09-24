import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
from pytorch_lightning import Trainer
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW, SGD, Adam
from pytorch_lightning.profiler import SimpleProfiler
from dataset import get_dataloader, get_dataloader_majority_vote
from models import GPT2Classifier, GPT2Classifier_reduced
from sklearn.metrics import precision_score, recall_score, f1_score
from params import configs


class TrainEngine(pl.LightningModule):
    def __init__(self, backbone, learning_rate):
        super().__init__()
        self.model = backbone
        self.learning_rate = learning_rate
        
    def forward(self, input_ids, attention_mask, rater_id, label=None):
        pass
            
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch

        out, loss = self.model(input_ids, attention_mask, label)
        correct_count = torch.sum(torch.argmax(label, dim=1) == torch.argmax(out, dim=1)).item()
        
        self.log('train_loss', loss, batch_size=configs.batch_size)
        return {"loss": loss, "correct_count": int(correct_count)}
            
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch

        out, loss = self.model(input_ids, attention_mask, label)
        correct_count = torch.sum(torch.argmax(label, dim=1) == torch.argmax(out, dim=1)).item()
        
        self.log('val_loss', loss, batch_size=configs.batch_size)
        return {"loss": loss, "correct_count": int(correct_count)}
        
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        
        out, loss = self.model(input_ids, attention_mask, label)
        correct_count = torch.sum(torch.argmax(label, dim=1) == torch.argmax(out, dim=1)).item()
        
        self.log('test_loss', loss, batch_size=configs.batch_size)
        return {"test_loss": loss, "outs": out.detach().cpu(), "labels": label.detach().cpu(), "correct_count": int(correct_count)}
    
    def training_step_end(self, outputs):
        return outputs
    
    def validation_step_end(self, outputs):
        return outputs
    
    def training_epoch_end(self, outputs):
        corrects = 0
        for out in outputs:
            corrects += out["correct_count"]
        self.log("train_epoch_acc", corrects/(len(outputs)*configs.batch_size))
    
    def validation_epoch_end(self, outputs):
        corrects = 0
        for out in outputs:
            corrects += out["correct_count"]
        self.log("val_epoch_acc", corrects/(len(outputs)*configs.batch_size))
    
    def test_epoch_end(self, outputs):
        outs = []
        labels = []
        
        precisions = {}
        recalls = {}
        f1s = {}
        
        for dic in outputs:
            outs.append(dic["outs"])
            labels.append(dic["labels"])
        
        outs = torch.cat(outs, dim=0)
        outputs = torch.zeros_like(outs)
        outputs.scatter_(1, torch.argmax(outs, dim=1).unsqueeze(1), 1)
        
        labels = torch.cat(labels, dim=0)
       
        label_to_emotion = {value: key for key, value in configs.six_emotion_label.items()}
        
        for label_idx in range(configs.num_classes):
            selected_idx = (torch.argmax(labels, axis=-1) == label_idx)
    
            class_pred = outputs[selected_idx]
            class_label = labels[selected_idx]
            
            precision = precision_score(class_pred[:,label_idx].int(), class_label[:,label_idx].int(), average='binary')
            recall = recall_score(class_pred[:,label_idx].int(), class_label[:,label_idx].int(), average='binary')
            f1 = f1_score(class_pred[:,label_idx].int(), class_label[:,label_idx].int(), average='binary')
            
            precisions[label_to_emotion[label_idx]] = precision
            recalls[label_to_emotion[label_idx]] = recall
            f1s[label_to_emotion[label_idx]] = f1
        
        for k, v in precisions.items():
            self.log(f"emotion {k} precision on test", v)
        for k, v in recalls.items():
            self.log(f"emotion {k} recall on test", v)
        for k, v in f1s.items():
            self.log(f"emotion {k} f1 on test", v)
        
        self.log("overall_test_precision", precision_score(outputs, labels, average='macro'))
        self.log("overall_test_recall", recall_score(outputs, labels, average='macro'))
        self.log("overall_test_f1", f1_score(outputs, labels, average='macro'))
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        default="train",
                        type=str,
                        required=False,
                        choices = ["train", "test"],
                        help="train mode or test mode, please first do train")
    parser.add_argument("--ckpt_path",
                        default=None,
                        type=str,
                        required=False,
                        help="path of ckpt")
    parser.add_argument("--log_path",
                        default="/projects/nie/goEmotion/logs/",
                        type=str,
                        required=False,
                        help="path of logs, ckpts, tensorboard files")
    
    args = parser.parse_args()
    
    
    check_point_callback = pl.callbacks.ModelCheckpoint(
        every_n_epochs=1,
        save_weights_only=True,
        monitor="val_epoch_acc",
        save_top_k=-1, 
        filename='{epoch:02d}-{val_epoch_acc:.2f}'
    )
    
    logger = TensorBoardLogger("/projects/nie/goEmotion/logs/", name="gpt2_related")
    profiler = SimpleProfiler(filename='profile')
    
    trainer = Trainer(
        fast_dev_run=False,
        gpus=[7],
        accelerator='gpu',
        max_epochs=configs.max_epoch_num,
        logger=logger,
        profiler=profiler,
        auto_lr_find=True,
        callbacks=[check_point_callback]
    )

    train_loader, val_loader, test_loader = get_dataloader()
    
    model = GPT2Classifier()
    
    train_engine = TrainEngine(backbone=model, learning_rate=configs.learning_rate)

    if args.mode in ["train"]:
        train_engine = TrainEngine(backbone=model, learning_rate=configs.learning_rate)
        trainer.fit(train_engine, train_loader, val_loader)
    
    if args.mode in ["test"]:
        train_engine = TrainEngine.load_from_checkpoint(args.ckpt_path, backbone=model, learning_rate=configs.learning_rate)
        trainer.test(train_engine, test_loader)

