#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/18 14:11
# @Author : kzl
# @Site : 
# @File : binary_bert.py

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
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, masked_softmax
from allennlp.training.metrics import F1Measure, Average, Metric, FBetaMeasure

import json
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.data.tokenizers import Tokenizer
import torch.nn.functional as F
from DataRead import *
import pickle

sym_size = 78
@DatasetReader.register("mds_reader")
class TextClassificationTxtReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 model: str = None) -> None:

        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._model = model

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields: Dict[str, Field] = {}
        # print(sample)
        tailored_history = sample['history']
        tailored_tags = sample['tags'][-10:]
        context = '。'.join(tailored_history)

        # history = ' '.join(list(''.join(context)))
        # context = '[CLS] ' + context[-512:]
        text_tokens = self._tokenizer.tokenize(context[-510:])
        fields['text'] = TextField(text_tokens, self._token_indexers)

        fileds_list = []
        for sen in tailored_history:
            sen = ' '.join(sen)
            txt_token = self._tokenizer.tokenize(sen)
            ff = TextField(txt_token, self._token_indexers)
            fileds_list.append(ff)
        fields["label"] = MultiLabelField(list(sample['next_symp']), skip_indexing=True, num_labels=sym_size)
        # fields['symptoms'] = MultiLabelField(list(sample['his_symp']), skip_indexing=True, num_labels=sym_size)
        # fields['tags'] = MetadataField(tailored_tags)
        # fields['history'] = ListField(fileds_list)
        fields["future"] = MultiLabelField(list(sample['future_symp']), skip_indexing=True, num_labels=sym_size)

        return Instance(fields)


@Model.register('symptoms_predictor')
class MyModel(Model):
    def __init__(self,
                text_field_embedder: TextFieldEmbedder,
                vocab: Vocabulary,
                seq2vec_encoder : Seq2VecEncoder = None,
                dropout: float = None,
                regularizer: RegularizerApplicator = None,
                ):
        super().__init__(vocab, regularizer)

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self.sym_size = sym_size
        self.embeddings = text_field_embedder
        self.vec_encoder = seq2vec_encoder
        self.hidden_dim = self.vec_encoder.get_output_dim()
        self.linear_class = torch.nn.Linear(self.hidden_dim, self.sym_size)
        # self.f_linear = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        with open('data/gcn_graph.pk','rb') as f:
            self.graph = torch.tensor(pickle.load(f)).cuda()

        self.topic_acc = Average()
        self.topic_rec = Average()
        self.topic_f1 = Average()
        self.macro_f = MacroF(self.sym_size)
        self.turn_acc = Average()
        # self.micro_f = FBetaMeasure(beta=1, average='micro')
        # self.macro_f = FBetaMeasure(beta=1, average='macro')

        self.future_acc = Average()

    def forward(self, text, label, **args):
        bs = label.size(0)
        embeddings = self.embeddings(text)  # bs  * seq_len * embedding

        mask = get_text_field_mask(text)  # bs * sen_num * sen_len

        seq_hidden = self.vec_encoder(embeddings, mask)  # bs , embedding

        topic_probs = F.sigmoid(self.linear_class(seq_hidden))
        # topic_weight = torch.ones_like(label) + 2 * label
        loss = F.binary_cross_entropy(topic_probs, label.float())
        output_dict = {'loss': loss}
        # _, max_index = torch.max(topic_probs, -1)
        pre_index = (topic_probs > 0.5).long()
        total_pre = torch.sum(pre_index)
        total_true = torch.sum(label)
        mask_index = (label == 1).long()
        self.macro_f(pre_index.cpu(), label.cpu())
        true_positive = (pre_index == label).long() * mask_index
        turn_true_num = (torch.sum(true_positive, 1) == torch.sum(mask_index, 1)).long()
        self.turn_acc(torch.sum(turn_true_num).item()/bs)
        pre_true = torch.sum(true_positive)
        # pre_true = torch.sum((pre_index == label).long() * mask_index)
        self.future_acc(torch.sum(pre_index * (args['future'] == 1).long()).item() / bs)

        acc,rec = 0., 0.
        if total_pre > 0:
            acc = (pre_true.float()/total_pre.float()).item()
        if total_true > 0:
            rec = (pre_true.float()/total_true.float()).item()
        self.topic_rec(rec)
        self.topic_acc(acc)
        return output_dict

    def get_metrics(self, reset=False):
        metrics = {}
        acc = self.topic_acc.get_metric(reset=reset)
        rec = self.topic_rec.get_metric(reset=reset)
        metrics['acc'] = acc
        metrics['rec'] = rec
        metrics['f1'] = 0
        if acc + rec > 0:
            metrics['f1'] = 2 * acc * rec / (acc + rec)
        metrics['macro_f1'] = self.macro_f.get_metric(reset=reset)
        metrics['turn_acc'] = self.turn_acc.get_metric(reset=reset)
        metrics['future_acc'] = self.future_acc.get_metric(reset=reset)
        return metrics
