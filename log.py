from dataclasses import dataclass
import datetime
import re
from typing import Optional

from .trainer import ModelTrainer

@dataclass
class ModelLogger:
    trainer: ModelTrainer
    print_to_stdout: bool = True
    print_to_file: Optional[str] = None
    print_time: Optional[bool] = False

    def __post_init__(self) -> None:
        """
        Perform the actions after the class instance is initialized
        """
        self.trainer.add_callback('on_before_start', self.on_before_start)
        self.trainer.add_callback('on_end_epoch', self.on_end_epoch)

    def on_before_start(self, **kwargs):
        """
        A callback that is called once before training

        :param kwargs: parameters passed to the callback function
        """
        summary = f'Start training for {kwargs['num_epochs']} epochs at {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}'
        self.output(summary)

    def on_end_epoch(self, **kwargs):
        """
        A callback that is called on the end of each of the training epochs

        :param kwargs: parameters passed to the callback function
        """
        result = kwargs['results'][kwargs['epoch']]
        timestr = f'\t{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}\t' if self.print_time else ''

        summary = f'{kwargs['epoch']}/{kwargs['num_epochs']}{timestr}\t\033[92mTrain {result['train']}\t\033[93mVal {result['validation']}\033[00m'
        self.output(summary)


    def output(self, summary:str) -> None:
        if self.print_to_stdout:
            print(summary)

        summary = re.sub('\033\[\d\dm', '', summary)

        if self.print_to_file:
            with open(self.print_to_file, 'a') as f:
                f.write(f'{summary}\n')
