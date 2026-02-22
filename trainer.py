from contextlib import nullcontext
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from tqdm import tqdm
from typing import Callable, Optional
import warnings

from .metrics import ModelMetrics


def device_auto(device: str='auto') -> torch.device:
    devices = {
        'cpu': 'cpu',
        'gpu': 'cuda',
        'cuda': 'cuda',
    }
    auto = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(devices.get(device, auto))


@dataclass
class SingleEpochTrainingResult:
    """
    Class storing the results of model training for a single epoch
    """
    predictions: np.ndarray
    targets: np.ndarray
    average_loss: float

    metrics: list = field(default_factory=list, init=False, repr=False)
    values: list = field(default_factory=list, init=False, repr=False)
    texts: list = field(default_factory=list, init=False, repr=True)

    def calculate_metrics(self, metrics: list[ModelMetrics], calculate: bool=True) -> None:
        """
        Calculate the metrics and clear the initial arrays to save RAM

        :param metrics: a list of metrics to calculate
        :param calculate: whether to actually perform a calculation or just erase the results
        """
        if not len(self.predictions):
            return

        self.metrics, self.str, self.val = [], [], []
        if calculate:
            for metric in metrics:
                metric_text, metric_value = metric(self.predictions, self.targets)
                if metric_value is not None:
                    self.metrics.append(metric.name)
                    self.values.append(metric_value)
                    self.texts.append(metric_text)

        # Clear the initial arrays
        self.predictions = np.array([])
        self.targets = np.array([])

    def __repr__(self) -> str:
        return f'loss={self.average_loss:.5f} {'\t'.join(self.texts)}'


@dataclass
class ModelTrainer:
    # Data loaders for train and validation sets
    train_loader: Optional[DataLoader] = None
    validation_loader: Optional[DataLoader] = None

    # How to train the model
    loss: Optional[Module] = None
    optimizer: Optional[Optimizer] = None
    scheduler: Optional[Scheduler] = None

    # Various options
    concatenate_results: bool = True  # Whether to concatenate predictions and targets before applying the metrics
    device: str | torch.device = 'auto'  # The device to use while training the model: cpu, gpu or auto
    limit_batches_per_epoch: int = 0  # if not zero, stop epoch on this value
    max_clip_grad_norm: float = 0  # Max norm of the gradients to be clipped; 0 for no clipping
    transfer_x_to_device: bool = True  # Whether to transfer data to `device` before applying the model
    use_tqdm: bool = True  # Whether to use tqdm
    validation_calculate_rate: int = 1  # 0 for never calculating validation metrics

    # What to calculate and what rate to use
    metrics: list[ModelMetrics] = field(default_factory=list)
    calculate_metrics_on_train: bool = False
    calculate_metrics_on_validation: bool = True

    # Callbacks with signature clb(trainer, model, epoch, results)
    on_before_start: list[Callable] = field(default_factory=list)
    on_start_epoch: list[Callable] = field(default_factory=list)
    on_before_train: list[Callable] = field(default_factory=list)
    on_after_train: list[Callable] = field(default_factory=list)
    on_before_validation: list[Callable] = field(default_factory=list)
    on_after_validation: list[Callable] = field(default_factory=list)
    on_end_epoch: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        """
        Perform the actions after the class instance is initialized
        """
        # Adjust the trainer's device
        if isinstance(self.device, str):
            devices = {'cpu': 'cpu', 'gpu': 'cuda', 'cuda': 'cuda'}
            auto = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(devices.get(self.device, auto))


    def _callback(self, event: str, args: dict) -> list:
        """
        Try to run callback functions if they exists

        :param event: the name of the event
        :param args: the arguments passed to the callback function
        :return: the result of the callback function is returned directly
        """
        callbacks = getattr(self, event, None)
        assert isinstance(callbacks, list), f'Callback `{event}()` does not exist'

        return [clb(**args) if callable(clb) else None for clb in callbacks]


    def add_callback(self, event: str, clb: Callable) -> None:
        """
        Add a callback function to the event specified by its name

        :param event: the name of the event
        :param clb: the callback function with a signature of clb(trainer, model, epoch, results)
        """
        callbacks = getattr(self, event, None)
        assert isinstance(callbacks, list), f'Callback `{event}()` does not exist'
        assert callable(clb), 'The object passed is not callable'
        callbacks.append(clb)


    def remove_callback(self, event: str, clb: Callable) -> None:
        """
        Removes the callback function from the event specified by its name

        :param event: the name of the event
        :param clb: the callback function to be removed
        """
        callbacks = getattr(self, event, None)
        assert isinstance(callbacks, list), f'Callback `{event}()` does not exist'
        assert callable(clb), 'The object passed is not callable'
        callbacks.remove(clb)


    def _process_epoch(self,
                       model: Module,
                       epoch: int,
                       loader: DataLoader,
                       training: bool = False,
                       scheduler: Optional[Scheduler] = None) -> SingleEpochTrainingResult:
        """
        Trains a model for a single epoch

        :param model: the model to train
        :param epoch: the current epoch number
        :param loader: the loader used to fetch the data
        :param training: whether to perform training on the provided data
        :param scheduler: the scheduler used for updating learning rate
        :return: the result of a single epoch training
        """
        accumulated_loss = 0
        accumulated_predictions = []
        accumulated_targets = []

        # Initial asserts
        assert self.loss is not None, '`loss` must not be None'
        assert self.optimizer is not None or not training, '`optimizer` must not be None if training is enabled'

        # We select the minibatches of objects one per step
        iterator = enumerate(tqdm(loader) if self.use_tqdm else loader)

        # Disable gradient calculations if no training is performed
        grad_calculation_context = nullcontext if training else torch.no_grad

        with grad_calculation_context():
            for i_step, (x, y) in iterator:
                # Limit batches number per epoch. If it is set, and we reach it, then stop
                if self.limit_batches_per_epoch and i_step >= self.limit_batches_per_epoch:
                    break

                # 1. Forward pass
                # 1.1. We calculate the model result on each minibatch
                x_gpu = x.to(self.device) if self.transfer_x_to_device else x
                prediction = model(x_gpu)

                # 1.2. We calculate the loss value
                y_gpu = y.to(self.device)
                loss_value = self.loss(prediction, y_gpu)
                assert not torch.isnan(loss_value), '`loss()` returned None'

                if training:
                    # 2. Backward pass is run only if training is enabled
                    # 2.1. We reset the gradient values ...
                    self.optimizer.zero_grad()

                    # 2.2. ... and calculate the new gradient values
                    loss_value.backward()

                    if self.max_clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_clip_grad_norm)

                    # 2.3. Then we renew the model parameters
                    self.optimizer.step()

                    # 2.4. We renew the scheduler step (if the scheduler is used)
                    if scheduler:
                        scheduler.step_update(num_updates=epoch * len(loader))

                accumulated_loss += loss_value.detach().cpu().numpy()
                accumulated_predictions.append(prediction.detach().cpu().numpy())
                accumulated_targets.append(y)

        # We make the scheduler step (if the scheduler is used)
        if training and scheduler:
            scheduler.step(epoch + 1)

        # We calculate the average loss and combine
        average_loss = accumulated_loss / (i_step + 1)

        if self.concatenate_results:
            accumulated_predictions = np.concatenate(accumulated_predictions, axis=0)
            accumulated_targets = np.concatenate(accumulated_targets)
        else:
            accumulated_predictions = np.array(accumulated_predictions)
            accumulated_targets = np.array(accumulated_targets)

        return SingleEpochTrainingResult(predictions=accumulated_predictions,
                                         targets=accumulated_targets,
                                         average_loss=average_loss)


    def __call__(self, model: Module, epochs: int | range) -> None:
        """
        Train the model

        :param model: the model to be trained
        :param epochs: the number of epochs or the particular range of epochs to use
        """
        # Cast `epochs` to range if it was provided as int
        assert isinstance(epochs, int) or isinstance(epochs, range), '`epochs` must be of int or range type'
        if isinstance(epochs, int):
            epochs = range(epochs)

        # Prepare for the main training loop
        results = {}

        # Main training loop
        self._callback(event='on_before_start', args={'trainer': self, 'model': model, 'epoch': None, 'num_epochs': len(epochs), 'results': results})
        for epoch in epochs:
            if epoch in results:
                warnings.warn(f'The epoch #{epoch} is already done. Skipping.')
                continue
            else:
                results[epoch] = {'epoch': epoch, 'train': None, 'validation': None}

            args = {'trainer': self, 'model': model, 'epoch': epoch, 'num_epochs': len(epochs), 'results': results}
            self._callback(event='on_start_epoch', args=args)

            self._callback(event='on_before_train', args=args)
            model.train()
            assert self.train_loader is not None, '`train_loader` must not be None'
            train_result = self._process_epoch(model=model,
                                               epoch=epoch,
                                               loader=self.train_loader,
                                               training=True,
                                               scheduler=self.scheduler)
            train_result.calculate_metrics(self.metrics, self.calculate_metrics_on_train)
            results[epoch]['train'] = train_result
            self._callback(event='on_after_train', args=args)

            # We validate the model (if it's time to do it)
            if (self.validation_calculate_rate > 0) and (epoch % self.validation_calculate_rate == 0):
                self._callback(event='on_before_validation', args=args)
                results[epoch]['validation'] = self.validate(model, epoch)
                self._callback(event='on_after_validation', args=args)

            self._callback(event='on_end_epoch', args=args)


    def validate(self, model: Module, epoch: int=0) -> SingleEpochTrainingResult:
        """
        Validate the model

        :param model: the model to validate
        """
        model.eval()
        assert self.validation_loader is not None, '`validation_loader` must not be None'
        validation_result = self._process_epoch(model=model,
                                                epoch=epoch,
                                                loader=self.validation_loader)
        validation_result.calculate_metrics(self.metrics, self.calculate_metrics_on_validation)
        return validation_result
