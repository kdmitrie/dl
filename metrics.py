from abc import ABC, abstractmethod
import numpy as np
from scipy.special import rel_entr, softmax
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple, Union


class ModelMetrics(ABC):
    """Base class representing metrics"""
    name: str = 'metrics'
    @abstractmethod
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float: pass

    def __call__(self,
                 predictions: np.ndarray,
                 targets: np.ndarray,
                 fmt: str = '.3f') -> Tuple[Union[None, str], Union[None, float]]:
        value = self.calc(predictions, targets)
        return ('{}={:'+fmt+'}').format(self.name, value), value


class ModelAccuracy(ModelMetrics):
    """Accuracy metrics"""
    name = 'acc'

    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predict_class = np.argmax(predictions, axis=-1)
        return sum(predict_class == targets) / len(targets)


class ModelROCAUC(ModelMetrics):
    """ROC AUC metrics variant taking into account only
    the classes, that exist in targets"""
    name = 'rocauc'
    average = 'macro'

    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        target_sums = targets.sum(axis=0)
        scored_columns = target_sums > 0
        return roc_auc_score(targets[:, scored_columns],
                             predictions[:, scored_columns],
                             average=self.average)


class ModelPaddedCMAP(ModelMetrics):
    """Padded CMAP macro metrics"""
    name = 'pcmap_score'
    average = 'macro'
    padding_factor = 5

    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        new_rows = np.ones((self.padding_factor, predictions.shape[-1]))
        targets = np.concatenate([targets, new_rows])
        predictions = np.concatenate([predictions, new_rows])
        score = average_precision_score(targets,
                                        predictions,
                                        average=self.average)
        return score


class ModelMultilabelAccuracy(ModelMetrics):
    """Multilabel Accuracy metrics"""
    name = 'mla'

    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predict_class = np.argmax(predictions, axis=-1)
        correct = [tgt[cl] for (cl, tgt) in zip(predict_class, targets)]
        return sum(correct) / len(targets)


class ModelKL(ModelMetrics):
    """Kullback–Leibler divergence metrics"""
    name = 'KL'
    epsilon = 10**-15
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
    """Class for storing and calculating the accuracy of model on each class"""
    name = 'class_accuracy'

    def __init__(self, num_classes=2, beta=0.9):
        self.num_classes = num_classes
        self.beta = beta
        self.probabilities = np.zeros(num_classes)
        self.totals = np.zeros(num_classes)

    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float: pass

    def get_probabilities(self):
        return self.probabilities

    def __call__(self,
                 predictions: np.ndarray,
                 targets: np.ndarray,
                 fmt: str = '.3f') -> Union[None, str]:
        preds = np.argmax(predictions, axis=1)

        # Total number of each class occurence
        totals = np.zeros(self.num_classes)
        keys_totals, count_totals = np.unique(targets, return_counts=True)
        totals[keys_totals] = count_totals

        # Number of each class correct results
        probabilities = np.zeros(self.num_classes)
        keys_correct, count_correct = np.unique(targets[targets == preds],
                                                return_counts=True)
        probabilities[keys_correct] = count_correct / totals[keys_correct]

        numer = self.probabilities * self.totals * self.beta + \
            probabilities * totals * (1 - self.beta)
        denom = self.totals * self.beta + totals * (1 - self.beta) + 1e-10
        self.probabilities = numer / denom

        self.totals += totals
        return None
