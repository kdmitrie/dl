from dataclasses import dataclass
import os
import sys
import torch
from typing import Any, Optional
import warnings

from .trainer import ModelTrainer

@dataclass
class ModelSaver:
    trainer: ModelTrainer
    save_model_path: str
    save_checkpoint_path: Optional[str] = None
    restore_checkpoint: bool = True
    random_seed: int = 20260106

    def __post_init__(self) -> None:
        """
        Perform the actions after the class instance is initialized
        """
        self.trainer.add_callback('on_before_start', self.on_before_start)
        self.trainer.add_callback('on_end_epoch', self.on_end_epoch)

        self.script_signature = ' '.join(sys.argv)

        if self.restore_checkpoint:
            if self.trainer.train_loader.generator is None:
                warnings.warn("""`train_loader` must have a particular generator specified. Otherwise, training is not deterministic""")
            if self.trainer.train_loader.num_workers <= 0:
                warnings.warn("""`train_loader` must have `num_workers` greater than 0. Otherwise, training is not deterministic""")

            if self.trainer.validation_loader.generator is None:
                warnings.warn('`validation_loader` must have a particular generator specified. Otherwise, validation is not deterministic')
            if self.trainer.validation_loader.num_workers <= 0:
                warnings.warn("""`validation_loader` must have `num_workers` greater than 0. Otherwise, validation is not deterministic""")


    def on_before_start(self, **kwargs):
        """
        A callback that is called before training

        :param kwargs: parameters passed to the callback function
        """
        if self.restore_checkpoint and self.save_checkpoint_path is not None and os.path.exists(self.save_checkpoint_path):
            checkpoint = torch.load(self.save_checkpoint_path, weights_only=False)

            if checkpoint['script_signature'] != self.script_signature:
                return

            kwargs['model'].load_state_dict(checkpoint['model'])

            if self.trainer.optimizer:
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer'])

                if self.trainer.scheduler:
                    self.trainer.scheduler.load_state_dict(checkpoint['scheduler'])
                    self.trainer.scheduler.step_update(num_updates=checkpoint['epoch'] * len(self.trainer.train_loader))
                    self.trainer.scheduler.step(checkpoint['epoch'] + 1)

            if self.trainer.train_loader.num_workers and self.trainer.train_loader.generator:
                self.trainer.train_loader.generator.set_state(checkpoint['train_loader_generator'])

            if self.trainer.validation_loader.num_workers and self.trainer.validation_loader.generator:
                self.trainer.validation_loader.generator.set_state(checkpoint['validation_loader_generator'])

            for k, v in checkpoint['results'].items():
                kwargs['results'][k] = v


    def on_end_epoch(self, **kwargs):
        """
        A callback that is called on the end of each of the training epochs

        :param kwargs: parameters passed to the callback function
        """
        epoch = kwargs['epoch']

        # Save the model
        self.save(kwargs['model'], self.save_model_path, epoch)

        # Save the training state
        if self.save_checkpoint_path is not None:
            checkpoint = {
                'epoch': epoch,
                'model': kwargs['model'].state_dict(),
                'scheduler': self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,
                'optimizer': self.trainer.optimizer.state_dict() if self.trainer.optimizer else None,
                'results': kwargs['results'],
                'script_signature': self.script_signature,
            }

            if self.trainer.train_loader.num_workers and self.trainer.train_loader.generator:
                checkpoint['train_loader_generator'] = self.trainer.train_loader.generator.get_state()
            if self.trainer.validation_loader.num_workers and self.trainer.validation_loader.generator:
                checkpoint['validation_loader_generator'] = self.trainer.validation_loader.generator.get_state()

            self.save(checkpoint, self.save_checkpoint_path, epoch)

    def save(self, object: Any, path: str, epoch: int) -> None:
        """
        Save the object under the specified path

        :param object: the object to save
        :param path: the path; if it contains %d, then epoch is substituted inside it
        :param epoch: the epoch number
        """
        fname = path % epoch if '%' in path else path
        tmpname = fname + '.tmp'
        torch.save(object, tmpname)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(tmpname, fname)
