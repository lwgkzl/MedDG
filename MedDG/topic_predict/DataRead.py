from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.fields import Field, TextField, MetadataField, MultiLabelField, ListField, LabelField
from typing import List, Dict
import tempfile
import torch
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.fields import TextField, MultiLabelField, ListField, Field, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import json
import pickle
import random
import sys


from overrides import overrides

from allennlp.training.metrics.metric import Metric



@Metric.register("MacroF1")
class MacroF(Metric):

    def __init__(self, num_class) -> None:
        self._true_positive = torch.zeros(num_class)
        self._predict = torch.zeros(num_class)
        self._ground_truth = torch.zeros(num_class)
        self._num_class = num_class

    @overrides
    def __call__(self, predict, ground_true):
        true_positive_index = (predict == ground_true).long() * (ground_true == 1).long()
        self._true_positive += torch.sum(true_positive_index, 0).squeeze(0)
        self._ground_truth += torch.sum(ground_true, 0).squeeze(0)
        self._predict += torch.sum(predict, 0).squeeze(0)


    @overrides
    def get_metric(self, reset: bool = False):
        precision = self._prf_divide(self._true_positive, self._predict)
        recall = self._prf_divide(self._true_positive, self._ground_truth)
        f_score = self._prf_divide(2.0 * precision * recall, (precision + recall))
        true_f_score = f_score[self._ground_truth>0]
        average_value = true_f_score.mean()
        if reset:
            self.reset()
        return average_value.item()

    @overrides
    def reset(self):
        self._true_positive = torch.zeros(self._num_class)
        self._predict = torch.zeros(self._num_class)
        self._ground_truth = torch.zeros(self._num_class)

    def _prf_divide(self, numerator1, denominator):
        result = numerator1 / denominator
        mask = denominator == 0.0
        if not mask.any():
            return result

        # remove nan
        result[mask] = 0.0
        return result