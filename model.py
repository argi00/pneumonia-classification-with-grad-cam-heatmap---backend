import torch.nn as nn
import lightning as pl
from torchvision import models
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.encoder_name == "densenet121":
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

    def forward(self, image: torch.Tensor):
        return self.model(image)


class ModelPumonie(pl.LightningModule):
    def __init__(self, args):
        super().__init__()  
        self.args = args

        # Initialisation des métriques
        self.train_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
        self.val_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
        self.test_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)

        self.val_precision = torchmetrics.Precision(task="binary", threshold=0.5)
        self.test_precision = torchmetrics.Precision(task="binary", threshold=0.5)

        self.val_recall = torchmetrics.Recall(task="binary", threshold=0.5)
        self.test_recall = torchmetrics.Recall(task="binary", threshold=0.5)

        self.val_f1 = torchmetrics.F1Score(task="binary", threshold=0.5)
        self.test_f1 = torchmetrics.F1Score(task="binary", threshold=0.5)

        self.val_auc_pr = torchmetrics.AveragePrecision(task="binary")
        self.test_auc_pr = torchmetrics.AveragePrecision(task="binary")

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32))

        # Initialisation du modèle de classification
        self.model = Classifier(args=args)

        # Initialisation des listes pour les prédictions et les étiquettes
        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def shared_step(self, batch, stage):

        images, labels = batch

        # Labels pour BCEWithLogitsLoss
        labels = labels.float().unsqueeze(1)

        outputs = self(images)

        loss = self.loss_fn(outputs, labels)

        # Probabilités
        probs = torch.sigmoid(outputs)

        # Classes 0/1
        preds = (probs >= 0.5).int()

        # Labels pour TorchMetrics
        metric_labels = labels.int()

        if stage == "train":

            self.train_accuracy(preds, metric_labels)

        elif stage == "val":

            self.val_accuracy(preds, metric_labels)
            self.val_precision(preds, metric_labels)
            self.val_recall(preds, metric_labels)
            self.val_f1(preds, metric_labels)

            # IMPORTANT : AveragePrecision reçoit les probabilités
            # et des labels int/long
            self.val_auc_pr(probs, metric_labels)

            self.val_preds.extend(
                preds.cpu().numpy()
            )

            self.val_labels.extend(
                metric_labels.cpu().numpy()
            )

        elif stage == "test":

            self.test_accuracy(preds, metric_labels)
            self.test_precision(preds, metric_labels)
            self.test_recall(preds, metric_labels)
            self.test_f1(preds, metric_labels)

            self.test_auc_pr(probs, metric_labels)

            self.test_preds.extend(
                preds.cpu().numpy()
            )

            self.test_labels.extend(
                metric_labels.cpu().numpy()
            )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        self.log("val_auc_pr", self.val_auc_pr, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True)
        self.log("test_precision", self.test_precision, on_epoch=True)
        self.log("test_recall", self.test_recall, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        self.log("test_auc_pr", self.test_auc_pr, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        cm = confusion_matrix(self.val_labels, self.val_preds)
        self.log_confusion_matrix(cm, stage="Validation")
        self.val_preds.clear()
        self.val_labels.clear()

    def on_test_epoch_end(self):
        cm = confusion_matrix(self.test_labels, self.test_preds)
        self.log_confusion_matrix(cm, stage="Test")
        self.test_preds.clear()
        self.test_labels.clear()

    def log_confusion_matrix(self, cm, stage):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"{stage} Confusion Matrix")
        
        # Log dans WandB
        wandb.log({f"{stage} Confusion Matrix": plt})
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer