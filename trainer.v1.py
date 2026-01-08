from dataclasses import dataclass, field
from numbers import Number
import numpy as np
import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Callable, Union

from .metrics import ModelMetrics


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
    calc_metrics_on_train: bool = False
    calc_metrics_on_val: bool = True

    # Whether to concatenate predictions and targets in single arrays before metrics
    concatenate_results: bool = True

    # Whether to transfer data on GPU before applying the model
    transfer_x_to_device: bool = True

    # Whether to switch between eval and train while training
    switch_eval_train: bool = True

    # Whether to use tqdm
    use_tqdm: bool = True

    # Max norm of the gradients to be clipped; 0 for no clipping
    max_clip_grad_norm: float = 0

    validation_calculate_rate: int = 1  # 0 for never calculating validation metrics
    print_rate: int = 1  # 0 for never printing metrics
    print_to_file: str = '' # file to print the summary of the model run
    limit_batches_per_epoch: int = 0  # if not zero, stop epoch on this value
    save_model_path: str = None  # Where to save a model
    rewrite_saved_model: bool = False
    save_state_dict: bool = False  # To save the state dict of the model or not
    pass_training: bool = False # Whether to do not train the model
    #pass_training_if_model_exists: Union[bool, Callable] = False # Don't train the model if checkpoint exists

    # Callbacks
    on_before_train: Optional[Callable] = None  # on_before_train(trainer, model, epoch)
    on_after_train: Optional[Callable] = None  # on_after_train(trainer, model, epoch, results)
    on_before_test: Optional[Callable] = None  # on_before_test(trainer, model, epoch)
    on_after_test: Optional[Callable] = None  # on_after_test(trainer, model, epoch, results)
    on_model_save: Optional[Callable] = None  # on_model_save(trainer, model, epoch) -> user_data
    on_model_load: Optional[Callable] = None  # on_model_load(trainer, model, epoch, user_data)

    def _forward_pass(self, model: Module, x: Tensor, y: Tensor) -> Tuple[np.ndarray, Tensor]:
        """Applies the model and returns an array of predictions and a tensor of loss values"""

        # 1. We calculate the model result on each minibatch
        x_gpu = x.to(self.device) if self.transfer_x_to_device else x
        prediction = model(x_gpu)

        # 2. We calculate the loss function
        y_gpu = y.to(self.device)
        loss_value = self.loss(prediction, y_gpu)
        assert not torch.isnan(loss_value)
        return prediction.detach().cpu().numpy(), loss_value

    def _backward_pass_train(self, model: Module, loss_value: Module) -> Tuple[List[Number], List[Number]]:
        """Trains the model and returns an array of loss values"""

        # 1. We reset the gradient values ...
        self.optimizer.zero_grad()

        # 2. ... and calculate the new gradient values
        loss_value.backward()

        if False:  # debug huge gradients here
            print(loss_value)
            grads = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
            grads = torch.cat(grads)
            mg = torch.max(torch.abs(grads))
            print(mg)
            #print(grads.shape, max(abs(grads)))


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

        if self.concatenate_results:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets)

        return all_predictions, all_targets, average_loss

    def validate_model(self, model):
        with torch.no_grad():
            val_predictions, val_targets, val_loss = self._train_epoch(model, 0, self.val_loader, None, self._backward_pass_no_train)
            return [metrics(val_predictions, val_targets)[1] for metrics in self.metrics]

    def load_checkpoint(self, epoch, model):
        # Load the model
        state_dict = torch.load(self.save_model_path % epoch, weights_only=False).state_dict()
        model.load_state_dict(state_dict)

        # Load the checkpoint
        checkpoint = torch.load(self.save_model_path % epoch + '.checkpoint', weights_only=False)
        if checkpoint['optimizer']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if 'results' in checkpoint:
            results = checkpoint['results']

        if 'user_data' in checkpoint:
            user_data = checkpoint['user_data']
            if callable(self.on_model_load):
                self.on_model_load(trainer=self, model=model, epoch=epoch, user_data=user_data)

        return results

    def train_with_checkpoints(self, model: Module, num_epochs: int = 10, chunk_size: int = 8) -> List[Dict]:
        for epoch in range(num_epochs):
            if not os.path.exists(self.save_model_path % epoch):
                break
        else:
            print('All models trained. Exiting.')
            exit()

        last_checkpoint = (epoch // chunk_size) * chunk_size
        print(f'Start from epoch {last_checkpoint}')
        result = self(model, num_epochs=last_checkpoint + chunk_size, start_epoch=last_checkpoint)
        if last_checkpoint + chunk_size >= num_epochs:
            return result

        print(f'Training of {chunk_size} epochs done. exiting.')
        exit()


    def __call__(self, model: Module, num_epochs: int = 10, start_epoch: int = 0) -> List[Dict]:
        """Model train on specified number of `num_epochs`"""
        val_loss = False
        results = []

        # Load the model, results and scheduler/optimizer state
        if start_epoch > 0:
            results = self.load_checkpoint(start_epoch - 1, model)

        for epoch in range(start_epoch, num_epochs):
            results.append({'epoch': epoch, 'train': None, 'val': None})

            do_train = True
            if self.pass_training:
                train_loss = -1
                train_metrics = 'PASSED'
                do_train = False

            if do_train:
                # We train the model
                if callable(self.on_before_train):
                    self.on_before_train(trainer=self, model=model, epoch=epoch)

                if self.switch_eval_train:
                    model.train()
                train_predictions, train_targets, train_loss = self._train_epoch(model, epoch, self.train_loader, self.scheduler, self._backward_pass_train)

                # Calculate metrics
                metric_results_str = []
                metric_results_val = []
                if self.calc_metrics_on_train:
                    for metrics in self.metrics:
                        metric_str, metric_val = metrics(train_predictions, train_targets)
                        if metric_val is not None:
                            metric_results_val.append(metric_val)
                            metric_results_str.append('Train_' +  metric_str)
                results[-1]['train'] = metric_results_val
                train_metrics = ', '.join(metric_results_str)

                if callable(self.on_after_train):
                    self.on_after_train(trainer=self, model=model, epoch=epoch, results=results)

            # We validate the model (if it's time to do it)
            if (self.validation_calculate_rate > 0) and (epoch % self.validation_calculate_rate == 0):
                if self.switch_eval_train:
                    model.eval()
                with torch.no_grad():
                    if callable(self.on_before_test):
                        self.on_before_test(trainer=self, model=model, epoch=epoch)
                    val_predictions, val_targets, val_loss = self._train_epoch(model, epoch, self.val_loader, None, self._backward_pass_no_train)

                    # Calculate metrics
                    metric_results_str = []
                    metric_results_val = []
                    if self.calc_metrics_on_val:
                        for metrics in self.metrics:
                            metric_str, metric_val = metrics(val_predictions, val_targets)
                            if metric_val is not None:
                                metric_results_val.append(metric_val)
                                metric_results_str.append('Val_' +  metric_str)
                    results[-1]['val'] = metric_results_val
                    val_metrics = ', '.join(metric_results_str)

                    if callable(self.on_after_test):
                        self.on_after_test(trainer=self, model=model, epoch=epoch, results=results)

            # We print summary (if it's time to do it)
            if (self.print_rate > 0) and (epoch % self.print_rate == 0):
                epoch_str = f'{epoch}/{num_epochs}'
                train_str = f'Train loss={train_loss:.3f} {train_metrics}'
                val_str = f'Val loss={val_loss:.3f} {val_metrics}' if val_loss else ''
                summary = '\t'.join([epoch_str, train_str, val_str])
                print(summary)

                if self.print_to_file != '':
                    with open(self.print_to_file, 'a') as f:
                        f.write(f'{summary}\n')

            # We save model if loss is not None and the path was specified
            if self.pass_training:
                continue

            if np.isnan(train_loss):
                return

            if (self.save_model_path is not None) and (self.rewrite_saved_model or not os.path.exists(self.save_model_path % epoch)):
                if callable(self.on_model_save):
                    user_data = self.on_model_save(trainer=self, model=model, epoch=epoch)
                else:
                    user_data = None

                # Save the model
                torch.save(model, self.save_model_path % epoch)

                # Save the training state
                checkpoint = {
                    'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                    'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                    'results': results,
                    'user_data': user_data,
                }
                torch.save(checkpoint, self.save_model_path % epoch + '.checkpoint')

                if self.save_state_dict:
                    torch.save(model.state_dict(), self.save_model_path % epoch + '.state-dict')

        return results
