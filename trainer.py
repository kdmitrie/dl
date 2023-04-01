from dataclasses import dataclass, field
from typing import List, Tuple, Callable
from numbers import Number

from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np


class ModelMetrics(ABC):
    """Base class representing metrics"""
    name: 'metrics'
    @abstractmethod
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float: pass

    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray, fmt:str = '.3f') -> str:
        value = self.calc(predictions, targets)
        return ('{}={:'+fmt+'}').format(self.name, value)


class ModelAccuracy(ModelMetrics):
    """Accuracy metrics"""
    name = 'acc'
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predict_class = np.argmax(predictions, axis=-1)
        return sum(predict_class == targets) / len(targets)
    
    
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
        
    validation_calculate_rate: int = 1 # 0 for never calculating validation metrics
    print_rate: int = 1 # 0 for never printing metrics
    limit_batches_per_epoch: int = 0 # if not zero, stop epoch on this value
    save_model_path: str = None # Where to save a model
    
    
    def _forward_pass(self, model: Module, x:Tensor, y:Tensor) -> Tuple[np.ndarray, Tensor]:
        """Applies the model and returns an array of predictions and a tensor of loss values"""

        # 1. We calculate the model result on each minibatch
        x_gpu = x.to(self.device)
        prediction = model(x_gpu)
        
        # 2. We calculate the loss function
        y_gpu = y.to(self.device)
        loss_value = self.loss(prediction, y_gpu)
        return prediction.detach().cpu().numpy(), loss_value
    

    def _backward_pass_train(self, loss_value: Module) -> Tuple[List[Number], List[Number]]:
        """Trains the model and returns an array of loss values"""

        # 1. We reset the gradient values ...
        self.optimizer.zero_grad()
        
        # 2. ... and calculate the new gradient values
        loss_value.backward()
        
        # 3. Then we renew the model parameters
        self.optimizer.step()
            
        return loss_value.detach().cpu().numpy()
    

    def _backward_pass_no_train(self, loss_value: Module) -> Tuple[List[Number], List[Number]]:
        """Returns the detached loss value without model training"""
        return loss_value.detach().cpu().numpy()
    
    
    def _train_epoch(self, model: Module, epoch: int, loader: DataLoader, scheduler: _LRScheduler, backward: Callable) -> Tuple[np.ndarray, np.ndarray, float]:
        """Trains a model for a single epoch"""
        num_updates = epoch * len(loader)

        loss_accum = 0
        all_predictions = []
        all_targets = []

        # We select the minibatches of objects one per step
        for i_step, (x, y) in enumerate(loader):
            # Limit batches number per epoch. If it is set, and we reach it, then stop
            if self.limit_batches_per_epoch and i_step >= self.limit_batches_per_epoch:
                break

            #print(f'\t\tStep {i_step}/{len(loader)}')
            train_prediction, train_loss = self._forward_pass(model, x, y)
            train_loss_value = backward(train_loss)
            
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

    
    
    def __call__(self, model: Module, num_epochs: int = 10) -> None:
        """Model train on specified number of `num_epochs`"""
        val_loss = False
        for epoch in range(num_epochs):
            # We train the model
            model.train()
            train_predictions, train_targets, train_loss = self._train_epoch(model, epoch, self.train_loader, self.scheduler, self._backward_pass_train)
            
            # Calculate metrics
            train_metrics = ', '.join('Train_' + metrics(train_predictions, train_targets) for metrics in self.metrics)
            
            # We validate the model (if it's time to do it)
            if (self.validation_calculate_rate > 0) and (epoch % self.validation_calculate_rate == 0):
                model.eval()
                with torch.no_grad():
                    val_predictions, val_targets, val_loss = self._train_epoch(model, epoch, self.val_loader, None, self._backward_pass_no_train)
                    # Calculate metrics
                    val_metrics = ', '.join('Val_' + metrics(val_predictions, val_targets) for metrics in self.metrics)

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
                torch.save(model, self.save_model_path % epoch)