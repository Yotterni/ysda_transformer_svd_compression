import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

import numpy as np

from sklearn.metrics import top_k_accuracy_score

from tqdm.auto import tqdm


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 device: torch.device,
                 ) -> None:
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.model.to(self.device)

    def train_step(self) -> list[float]:
        loss_history = []
        progress_bar = tqdm(self.train_loader)
        for batch in progress_bar:
            x, labels = batch
            x, labels = x.to(self.device), labels.to(self.device)
            predictions = self.model(X).logits

            loss = self.loss_function(predictions, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            progress_bar.set_postfix(train_loss=loss.item())

        return loss_history

    @torch.no_grad()
    def val_step(self,
                 k_for_top_k: int = 5,
                 first_m_batches: int | None = None
                 ) -> dict[str, np.floating]:
        acc_evaluator = Accuracy(task='multiclass', num_classes=1000)

        accuracies = []
        top_k_accuracies = []
        loss_history = []

        progress_bar = tqdm(self.val_loader)
        for idx, batch in enumerate(progress_bar):
            if first_m_batches is not None and idx + 1 == first_m_batches:
                break

            x, labels = batch
            x = x.to(self.device)
            logits = self.model(x).logits.detach().cpu()

            loss = self.loss_function(logits, labels)
            loss_history.append(loss.item())
            progress_bar.set_postfix(val_loss=loss.item())

            accuracy = acc_evaluator(logits.argmax(dim=1), labels)
            # print(labels, logits.argmax(dim=1).numpy())
            top_k_accuracy = top_k_accuracy_score(
                labels.numpy(), logits.numpy(), labels=np.arange(0, 1000))

            accuracies.append(accuracy)
            top_k_accuracies.append(top_k_accuracy)

        return {
            'accuracy': np.mean(accuracies),
            f'accuracy@{k_for_top_k}': np.mean(top_k_accuracies),
            'loss': np.mean(loss_history)
        }

    def run(self, num_epochs: int) -> tuple[list[float], list[float]]:
        train_history = []
        val_history = []
        for _ in tqdm(range(num_epochs)):
            train_history.extend(self.train_step())
            val_history.extend(self.val_step())

        return train_history, val_history