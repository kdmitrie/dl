import glob
from dataclasses import dataclass, field
import os
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .metrics import ModelMetrics
from .trainer import ModelTrainer, SingleEpochTrainingResult

@dataclass
class ModelSoupGreedy:
    trainer: ModelTrainer
    save_model_path: str
    save_checkpoint_path: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    metrics_direction: str = 'minimize'
    loader: DataLoader | str = 'validation'
    results: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """
        Perform the actions after the class instance is initialized
        """
        model_scores = self.get_initial_scores()
        self.create_soup(model_scores)


    def get_initial_scores(self) -> list[tuple[str, float]]:
        """
        For each available model, gets its score. Uses available results from training, if it's posible.

        :return: a list of tuples containing path to model and its corresponding score
        """

        # Get model paths
        assert '%' in self.save_model_path, '`save_model_path` must contain %i for epoch number to create the soup'
        model_paths = glob.glob(self.save_model_path.replace('%i', '*'))
        assert len(model_paths) > 0, f'No models were found at {self.save_model_path}'

        # Load results if they exist
        if self.save_checkpoint_path is not None and os.path.exists(self.save_checkpoint_path):
            self.results = torch.load(self.save_checkpoint_path, weights_only=False)['results']

        # For each model, get its score
        model_scores = []
        for model_path in model_paths:
            score = self.load_score_from_results(model_path)
            if score is None:
                model = torch.load(model_path, weights_only=False)
                score = self.get_model_score(model)
            model_scores.append((model_path, score))
        return model_scores


    def create_soup(self, model_scores):
        model_scores = sorted(model_scores, key=lambda x: x[1])

        # Load current best model
        model_path, best_score = model_scores[0]
        models_in_soup = [model_path]
        best_model = torch.load(model_path, weights_only=False)

        # The direction of metrics optimization
        compare = lambda a, b: a < b if self.metrics_direction == 'minimize' else b > a

        print(f'Create SOUP. Initial model {model_path}({best_score})')

        # Try to create a soup using greedy strategy
        for model_path, model_score in model_scores[1:]:
            model = torch.load(model_path, weights_only=False)
            for p1, p2 in zip(model.parameters(), best_model.parameters()):
                p1.data = (p1.data + p2.data * len(models_in_soup)) / (len(models_in_soup) + 1)
            new_score = self.get_model_score(model)

            if compare(new_score, best_score):
                print(f'Add model {model_path}({model_score}) => soup({new_score})')
                models_in_soup.append(model_path)
                best_model = model
                best_score = new_score

        return best_model, best_score, models_in_soup


    def get_model_score(self, model: torch.nn.Module) -> float:
        """
        Validates the model by actually running it accross all the provided samples and returns its score

        :param model: the model to validate
        :return: the score of the model
        """
        loaders = {'train': self.trainer.train_loader, 'validation': self.trainer.validation_loader}
        loader = loaders.get(self.loader, self.loader)
        result = self.trainer.validate(model, loader, None if self.metrics is None else [self.metrics])
        return self._get_result_metrics(result)


    def load_score_from_results(self, model_path: str) -> Optional[float]:
        """
        Tries to use training results to get the model's score. Returns None on failure.

        :param model_path: the path to the model to validate
        :return: the model's score or None if it was not found
        """
        if not (match := re.match(re.escape(self.save_model_path).replace('%i','(\\d+)'), model_path)):
            return None
        if (epoch := int(match.group(1))) not in self.results:
            return None
        if self.loader in ['train', 'validation']:
            result = self.results[epoch][self.loader]
            return self._get_result_metrics(result)


    def _get_result_metrics(self, result: SingleEpochTrainingResult) -> Optional[float]:
        """
        Returns the score of the model based on its validation results and provided metrics

        :param result: the results of model validation
        :return: the model score or None if it was not found
        """
        if self.metrics is None:
            return result.average_loss
        elif self.metrics.name in result.metrics:
            metrics_index = result.metrics.index(self.metrics.name)
            return result.values[metrics_index]

