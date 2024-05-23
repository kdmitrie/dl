from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Union
from numbers import Number

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
from scipy.special import rel_entr, softmax
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

class ModelMetrics(ABC):
    """Base class representing metrics"""
    name: 'metrics'
    @abstractmethod
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float: pass

    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray, fmt:str = '.3f') -> Union[None, str]:
        value = self.calc(predictions, targets)
        return ('{}={:'+fmt+'}').format(self.name, value)

    

class ModelAccuracy(ModelMetrics):
    """Accuracy metrics"""
    name = 'acc'
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predict_class = np.argmax(predictions, axis=-1)
        return sum(predict_class == targets) / len(targets)

    

class ModelROCAUC(ModelMetrics):
    """ROC AUC metrics variant taking into account only the classes, which exist in targets"""
    name = 'rocauc'
    average = 'macro'
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        target_sums = targets.sum(axis=0)
        scored_columns = target_sums > 0
        return roc_auc_score(targets[:, scored_columns], predictions[:, scored_columns], average=self.average)
        


class ModelMultilabelAccuracy(ModelMetrics):
    """Multilabel Accuracy metrics"""
    name = 'mla'
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predict_class = np.argmax(predictions, axis=-1)
        return sum(tgt[cl] for (cl, tgt) in zip(predict_class, targets)) / len(targets)


class ModelKL(ModelMetrics):
    """Kullback–Leibler divergence metrics"""
    name = 'KL'
    epsilon = float=10**-15
    processing = 'softmax'
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        if self.processing == 'exp':
            probabilities = np.exp(predictions)
        elif self.processing == 'softmax':
            probabilities = softmax(predictions, axis=1)
        else:
            probabilities = predictions
        submission = np.clip(probabilities, self.epsilon, 1 - self.epsilon)
        entropy = rel_entr(targets, submission)
        return np.average(entropy.sum(axis=1))
    
    
class ModelStatisticsClassAccuracy(ModelMetrics):
    """Class for storing and calculating the accuracy of a model on each class"""
    name = 'class_accuracy'

    def __init__(self, num_classes=2, beta=0.9):
        self.num_classes = num_classes
        self.beta = beta
        self.probabilities = np.zeros(num_classes)
        self.totals = np.zeros(num_classes)


    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float: pass


    def get_probabilities(self):
        return self.probabilities

    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray, fmt:str = '.3f') -> Union[None, str]:
        preds = np.argmax(predictions, axis = 1)

        # Total number of each class occurence
        totals = np.zeros(self.num_classes)
        keys_totals, count_totals = np.unique(targets, return_counts=True)
        totals[keys_totals] = count_totals

        # Number of each class correct results
        probabilities = np.zeros(self.num_classes)
        keys_correct, count_correct = np.unique(targets[targets == preds], return_counts=True)
        probabilities[keys_correct] = count_correct / totals[keys_correct]
        
        self.probabilities = (self.probabilities * self.totals * self.beta + probabilities * totals * (1 - self.beta)) / (self.totals * self.beta + totals * (1 - self.beta) + 1e-10)

        self.totals += totals
        return None
    
    
    
@dataclass
class ModelTrainer:
    """Class allows to set the parameters and train the models"""
    # Data loaders for train and validation sets
    train_loader: DataLoader
    val_loader: DataLoader
        
    # How to train the model
    loss: Module
    optimizer: Optimizer
    scheduler: _LRScheduler = None
    
    # Where to train the model
    device: str = 'cpu'
        
    # What to calculate and what rate to use
    metrics: List[ModelMetrics] = field(default_factory=list)
        
    # Whether to transfer data on GPU before applying the model
    transfer_x_to_device: bool = True
        
    # Whether to switch between eval and train while training
    switch_eval_train: bool = True

    # Whether to use tqdm
    use_tqdm: bool = True

    # Max norm of the gradients to be clipped; 0 for no clipping
    max_clip_grad_norm: float = 0
        
    validation_calculate_rate: int = 1 # 0 for never calculating validation metrics
    print_rate: int = 1 # 0 for never printing metrics
    limit_batches_per_epoch: int = 0 # if not zero, stop epoch on this value
    save_model_path: str = None # Where to save a model
        
    # Callbacks
    onModelSave : Union[Callable, str] = None # onModelSave(trainer, model, epoch, file)
    
    def _forward_pass(self, model: Module, x:Tensor, y:Tensor) -> Tuple[np.ndarray, Tensor]:
        """Applies the model and returns an array of predictions and a tensor of loss values"""

        # 1. We calculate the model result on each minibatch
        x_gpu = x.to(self.device) if self.transfer_x_to_device else x
        prediction = model(x_gpu)
        
        # 2. We calculate the loss function
        y_gpu = y.to(self.device)
        loss_value = self.loss(prediction, y_gpu)
        return prediction.detach().cpu().numpy(), loss_value
    

    def _backward_pass_train(self, model: Module, loss_value: Module) -> Tuple[List[Number], List[Number]]:
        """Trains the model and returns an array of loss values"""

        # 1. We reset the gradient values ...
        self.optimizer.zero_grad()
        
        # 2. ... and calculate the new gradient values
        loss_value.backward()

        if self.max_clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_clip_grad_norm)
        
        # 3. Then we renew the model parameters
        self.optimizer.step()
            
        return loss_value.detach().cpu().numpy()
    

    def _backward_pass_no_train(self, model: Module, loss_value: Module) -> Tuple[List[Number], List[Number]]:
        """Returns the detached loss value without model training"""
        return loss_value.detach().cpu().numpy()
    
    
    def _train_epoch(self, model: Module, epoch: int, loader: DataLoader, scheduler: _LRScheduler, backward: Callable) -> Tuple[np.ndarray, np.ndarray, float]:
        """Trains a model for a single epoch"""
        num_updates = epoch * len(loader)

        loss_accum = 0
        all_predictions = []
        all_targets = []

        # We select the minibatches of objects one per step
        iterator = enumerate(tqdm(loader)) if self.use_tqdm else enumerate(loader)
        for i_step, (x, y) in iterator:
            # Limit batches number per epoch. If it is set, and we reach it, then stop
            if self.limit_batches_per_epoch and i_step >= self.limit_batches_per_epoch:
                break

            #print(f'\t\tStep {i_step}/{len(loader)}')
            train_prediction, train_loss = self._forward_pass(model, x, y)
            train_loss_value = backward(model, train_loss)
            
            # We renew the scheduler step (if the scheduler is used)
            if scheduler:
                scheduler.step_update(num_updates=num_updates)

            loss_accum += train_loss_value
            all_predictions.append(train_prediction)
            all_targets.append(y)
            

        # We make the scheduler step (if the scheduler is used)
        if scheduler:
            scheduler.step(epoch + 1)
            
        # We calculate the metrics on train and on validation sets
        average_loss = loss_accum / (i_step + 1)
        
        return np.concatenate(all_predictions, axis=0), np.concatenate(all_targets), average_loss

    
    def validate_model(self, model):
        with torch.no_grad():
            val_predictions, val_targets, val_loss = self._train_epoch(model, 0, self.val_loader, None, self._backward_pass_no_train)
            val_metrics = ', '.join('Val_' + metrics(val_predictions, val_targets) for metrics in self.metrics)
            print(f'Val loss={val_loss:.3f} {val_metrics}')
        
        
    def __call__(self, model: Module, num_epochs: int = 10) -> None:
        """Model train on specified number of `num_epochs`"""
        val_loss = False
        for epoch in range(num_epochs):
            # We train the model
            if self.switch_eval_train:
                model.train()
            train_predictions, train_targets, train_loss = self._train_epoch(model, epoch, self.train_loader, self.scheduler, self._backward_pass_train)
            
            # Calculate metrics
            metric_results = []
            for metrics in self.metrics:
                metric_result = metrics(train_predictions, train_targets)
                if metric_result is not None:
                    metric_results.append('Train_' +  metric_result)
            train_metrics = ', '.join(metric_results)
            
            # We validate the model (if it's time to do it)
            if (self.validation_calculate_rate > 0) and (epoch % self.validation_calculate_rate == 0):
                if self.switch_eval_train:
                    model.eval()
                with torch.no_grad():
                    val_predictions, val_targets, val_loss = self._train_epoch(model, epoch, self.val_loader, None, self._backward_pass_no_train)
                    # Calculate metrics
                    metric_results = []
                    for metrics in self.metrics:
                        metric_result = metrics(val_predictions, val_targets)
                        if metric_result is not None:
                            metric_results.append('Val_' +  metric_result)
                    val_metrics = ', '.join(metric_results)

            # We print summary (if it's time to do it)
            if (self.print_rate > 0) and (epoch % self.print_rate == 0):
                epoch_str = f'{epoch}/{num_epochs}'
                train_str = f'Train loss={train_loss:.3f} {train_metrics}'
                val_str = f'Val loss={val_loss:.3f} {val_metrics}' if val_loss else ''
                print('\t'.join([epoch_str, train_str, val_str]))
                
            # We save model if loss is not None and the path was specified
            if np.isnan(train_loss):
                return
            if (self.save_model_path is not None):
                torch.save(model.cpu(), self.save_model_path % epoch)
                model.to(self.device)
                
                if callable(self.onModelSave):
                    self.onModelSave(trainer=self, model=model, epoch=epoch, file=self.save_model_path % epoch)
