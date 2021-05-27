from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField ,ListField, MultiLabelField
from allennlp.data.instance import Instance
import pickle
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from typing import Tuple, Dict, Optional
from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set,List

from overrides import overrides
import torch
from allennlp.training.metrics.metric import Metric
import random
import pkuseg

import copy


total_entiy = 160
@DatasetReader.register("seqreader_word")
class Seq2SeqDatasetReader(DatasetReader):

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = 510,
        target_max_tokens: Optional[int] = 64,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.pre_sen = 10
        # with open('../data/0831/total_bert_hidden.pk','rb') as f:
        #     self.bert_hidden = pickle.load(f)

    @overrides
    def _read(self, file_path: str):
        with open('../data/160_last_topic2num.pk','rb') as f:
            topic2num = pickle.load(f)
        num2topic = {k: v for v, k in topic2num.items()}
        # new_dialog.append({"history": copy.deepcopy(history), "next_sym": copy.deepcopy(xz),
        #                               'response': sen['sentence'], 'id': start_index, 'history_with_topic': copy.deepcopy(history_with)})
        with open(file_path, 'rb') as f:
            new_dialog = pickle.load(f)
            for dic in new_dialog:
                if 'train' in file_path:
                    dic['history'][-1] = dic['history'][-1] + ''.join([num2topic[x] for x in dic['next_sym']]) + ":"
                else:
                    dic['history'][-1] = dic['history'][-1] + ''.join(dic['bert_word']) + ":"
                        # print(result['next_sym'])
                yield self.text_to_instance(dic)
        # print(file_path, start_index)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields: Dict[str, Field] = {}
        sen_num = self.pre_sen
        context = ' '.join(sample['history'])   # 带word的还可以考虑一下用sample['history_with_topic']
        all_sentence = sample['history']
        # history = ' '.join(list(''.join(context)))
        history = ' '.join(context)

        text_tokens = self._source_tokenizer.tokenize(history)
        text_tokens = text_tokens[-self._source_max_tokens:]
        text_tokens.insert(0, Token(START_SYMBOL))
        text_tokens.append(Token(END_SYMBOL))
        if random.random() < 0.0001:
            print("text_tokens", text_tokens)

        fileds_list = []
        for sen in all_sentence:
            sen = ' '.join(sen)
            txt_token = self._source_tokenizer.tokenize(sen)
            ff = TextField(txt_token, self._source_token_indexers)
            fileds_list.append(ff)
        # response = ' '.join(sample['response'])
        response = ' '.join(sample['response'])
        response_tokens = self._target_tokenizer.tokenize(response)
        response_tokens = response_tokens[:self._target_max_tokens]
        response_tokens.insert(0, Token(START_SYMBOL))
        response_tokens.append(Token(END_SYMBOL))
        fields['history'] = ListField(fileds_list)
        fields['id'] = MetadataField(sample['id'])
        fields['source_tokens'] = TextField(text_tokens, self._source_token_indexers)
        fields["next_sym"] = MultiLabelField(list(sample['next_sym']), skip_indexing=True, num_labels=total_entiy)
        fields['target_tokens'] = TextField(response_tokens, self._target_token_indexers)
        return Instance(fields)

@DatasetReader.register("seqreader")
class Seq2SeqDatasetReader(DatasetReader):

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = 510,
        target_max_tokens: Optional[int] = 64,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.pre_sen = 10

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            new_dialog = pickle.load(f)
            for dic in new_dialog:
                yield self.text_to_instance(dic)
        # print(file_path, start_index)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields: Dict[str, Field] = {}
        sen_num = self.pre_sen
        context = ' '.join(sample['history'])
        all_sentence = sample['history']
        # history = ' '.join(list(''.join(context)))
        history = ' '.join(context)

        text_tokens = self._source_tokenizer.tokenize(history)
        text_tokens = text_tokens[-self._source_max_tokens:]
        text_tokens.insert(0, Token(START_SYMBOL))
        text_tokens.append(Token(END_SYMBOL))
        if random.random() < 0.0001:
            print("text_tokens", text_tokens)
        fileds_list = []
        for sen in all_sentence:
            sen = ' '.join(sen)
            txt_token = self._source_tokenizer.tokenize(sen)
            ff = TextField(txt_token, self._source_token_indexers)
            fileds_list.append(ff)

        response = ' '.join(sample['response'])
        response_tokens = self._target_tokenizer.tokenize(response)
        response_tokens = response_tokens[:self._target_max_tokens]
        response_tokens.insert(0, Token(START_SYMBOL))
        response_tokens.append(Token(END_SYMBOL))
        # tailored_history = sample['history']
        fields['history'] = ListField(fileds_list)
        fields['id'] = MetadataField(sample['id'])
        fields['source_tokens'] = TextField(text_tokens, self._source_token_indexers)
        fields["next_sym"] = MultiLabelField(list(sample['next_sym']), skip_indexing=True, num_labels=total_entiy)
        fields['target_tokens'] = TextField(response_tokens, self._target_token_indexers)
        return Instance(fields)


from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import  re
def get_youyin(sentence):
    aa = "(吃早饭|吃饭规律|三餐规律|饮食规律|作息规律|辣|油炸|熬夜|豆类|油腻|生冷|煎炸|浓茶|喝酒|抽烟|吃的多|暴饮暴食|不容易消化的食物|情绪稳定|精神紧张|夜宵).*?(吗|啊？|呢？|么|嘛？)"
    bb = "吃得过多过饱|(有没有吃|最近吃|喜欢吃|经常进食|经常吃).*?(辣|油炸|油腻|生冷|煎炸|豆类|不容易消化的食物|夜宵)"
    cc = "(工作压力|精神压力|压力).*?(大不大|大吗)|(心情|情绪|精神).*(怎么样|怎样|如何)|(活动量|运动|锻炼).*?(大|多|少|多不多|咋样|怎么样|怎样).*?(吗|呢)"
    if re.search(aa,sentence) or re.search(bb,sentence) or re.search(cc,sentence):
        return True
    return False

def get_location(sentence):
    cc = '哪个部位|哪个位置|哪里痛|什么部位|什么位置|哪个部位|哪个位置|哪一块|那个部位痛|肚脐眼以上|描述一下位置|具体部位|具体位置'
    if re.search(cc,sentence) is not None:
        return True
    return False

def get_xingzhi(sentence):
    cc = '是哪种疼|怎么样的疼|绞痛|钝痛|隐痛|胀痛|隐隐作痛|疼痛.*性质|(性质|什么样).*(的|得)(痛|疼)'
    if re.search(cc,sentence) is not None:
        return True
    return False

def get_fan(sentence):
    cc = '(饭.*?前|吃东西前|餐前).*(疼|痛|不舒服|不适)|(饭.*?后|吃东西后|餐后).*(疼|痛|不舒服|不适)|(早上|早晨|夜里|半夜|晚饭).*(疼|痛|不舒服|不适)'
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        if aa[1] - aa[0] <20:
            return True
    return False

def get_tong_pinglv(sentence):
    cc = '持续的疼|疼一会儿会自行缓解|持续的，还是阵发|症状减轻了没|(疼|痛).*轻|现在没有症状了吗|现在还有症状吗|(一阵一阵|一直|持续).*(疼|痛)|一阵阵.*(痛|疼)|阵发性|持续性'
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        return True
    return False

def get_tong(sentence):
    if get_tong_pinglv(sentence) or get_fan(sentence) or get_xingzhi(sentence):
        return True
    return False


def get_other_sym(sentence):
    cc = '(还有什么|还有啥|有没有其|都有啥|都有什么|还有别的|有其他|有没有什么|还有其他).*(症状|不舒服)|别的不舒服|有其他不舒服|主要是什么症状|主要.*症状|哪些不适症状|哪些.*症状|出现了什么症状'
    if re.search(cc,sentence) is not None:
        aa = re.search(cc,sentence).span()
        return True
    return False

def get_time(sentence):
    aa = "(情况|症状|痛|发病|病|感觉|疼|这样|不舒服|大约).*?(多久|多长时间|几周了？|几天了？)"
    bb = "(，|。|、|？)(多长时间了|多久了|有多久了|有多长时间了)|^(多久了|多长时间了|有多久了|有多长时间了|几天了|几周了)"
    cc = "有多长时间|有多久"
#     match_need = "多久了？|多长时间了？|几天了？|几周了？|多久了|多长时间了|几天了|几周了"
    if re.search(aa,sentence) is not None or re.search(bb,sentence) is not None:
        return True
    return False

@Metric.register("knowledge")
class KD_Metric(Metric):
    def __init__(self) -> None:
        self._pred_true = 0
        self._total_pred = 0
        self._total_true = 0
        with open('../data/new_cy_bii.pk', 'rb') as f:
            self.norm_dict = pickle.load(f)

    def reset(self) -> None:
        self._pred_true = 0
        self._total_pred = 0
        self._total_true = 0

    @overrides
    def get_metric(self, reset: bool = False):
        rec, acc, f1 = 0., 0., 0.
        # print("pred_true",self._pred_true)
        # print("_total_pred",self._total_pred)
        # print("_total_true",self._total_true)
        if self._total_pred > 0:
            acc = self._pred_true / self._total_pred
        if self._total_true > 0:
            rec = self._pred_true / self._total_true
        if acc > 0 and rec > 0:
            f1 = acc * rec * 2 / (acc + rec)
        if reset:
            self.reset()
        return {'rec':rec, 'acc':acc, 'f1':f1}

    def convert_sen_to_entity_set(self, sen):
        entity_set = set()
        for entity in self.norm_dict.keys():
            if entity in sen:
                entity_set.add(self.norm_dict[entity])
        if get_location(sen):
            entity_set.add('位置')
        if get_youyin(sen):
            entity_set.add('诱因')
        if get_tong(sen):
            entity_set.add('性质')
        if get_time(sen):
            entity_set.add('时长')
        return entity_set

    @overrides
    def __call__(
        self,
        references, # list(list(str))
        hypothesis, # list(list(str))
    ) -> None:
        # print("len: ",len(references))
        for batch_num in range(len(references)):
            ref = ''.join(references[batch_num])
            hypo = ''.join(hypothesis[batch_num])
            hypo_list = self.convert_sen_to_entity_set(hypo)
            # hypo_list = hypothesis[batch_num].split('.')
            ref_list = self.convert_sen_to_entity_set(ref)

            self._total_true += len(ref_list)
            self._total_pred += len(hypo_list)
            for entity in hypo_list:
                if entity in ref_list:
                    self._pred_true += 1


@Metric.register("nltk_bleu")
class NLTK_BLEU(Metric):
    def __init__(
        self,
        ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        self._ngram_weights = ngram_weights
        self._scores = []
        self.smoothfunc = SmoothingFunction().method7
        # if all(ngram_weights = SmoothingFunction().method0

    def reset(self) -> None:
        self._scores = []

    @overrides
    def get_metric(self, reset: bool = False):
        score = 0.
        if len(self._scores):
            score = sum(self._scores) / len(self._scores)
        if reset:
            self.reset()
        return score

    @overrides
    def __call__(
        self,
        references, # list(list(str))
        hypothesis, # list(list(str))
    ) -> None:
        for batch_num in range(len(references)):
            if len(hypothesis[batch_num]) <= 4:
                self._scores.append(0)
            else:
                self._scores.append(sentence_bleu([references[batch_num]], hypothesis[batch_num],
                                          smoothing_function=self.smoothfunc,
                                          weights=self._ngram_weights))

def my_sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None,
                                      ) -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.float()
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.log(logits_flat + 1e-16)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        return per_batch_loss


@Metric.register("my_average")
class MyAverage(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value, num):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value += list(self.unwrap_to_tensors(value))[0]
        self._count += num

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0


@Metric.register("my_F1")
class F1(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0

    @overrides
    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value = list(self.unwrap_to_tensors(value))[0]

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0

@Metric.register("distinct1")
class Distinct1(Metric):
    def __init__(self):
        self._total_vocabs = 0
        self.appear_vocabs = set()
    @overrides
    def __call__(self, hypothesis):
        batch_size = len(hypothesis)
        for b in range(batch_size):
            self._total_vocabs += len(hypothesis[b])
            self.appear_vocabs.update(hypothesis[b])

    def reset(self) -> None:
        print("-------------------------------")
        print("-"*100)
        # print(self._total_vocabs)
        # print(self.appear_vocabs)
        self._total_vocabs = 0
        self.appear_vocabs = set()

    def get_metric(self, reset: bool = False):
        # print("total_vocab", self._total_vocabs)
        if self._total_vocabs == 0:
            value = 0
        else:
            value = len(self.appear_vocabs) / self._total_vocabs
        if reset:
            self.reset()
        return value

@Metric.register("distinct2")
class Distinct2(Metric):
    def __init__(self):
        self._total_vocabs = 0
        self.appear_vocabs = set()

    @overrides
    def __call__(self, hypothesis):
        batch_size = len(hypothesis)
        for b in range(batch_size):
            if len(hypothesis[b]) <= 1:
                continue
            self._total_vocabs += len(hypothesis[b]) - 1
            for i in range(len(hypothesis[b])-1):
                self.appear_vocabs.add(hypothesis[b][i]+hypothesis[b][i+1])

    def reset(self) -> None:
        # print("-------------------------------")
        # print("-"*1000)
        # print(self._total_vocabs)
        # print(self.appear_vocabs)

        self._total_vocabs = 0
        self.appear_vocabs = set()

    def get_metric(self, reset: bool = False):
        if self._total_vocabs == 0:
            value = 0
        else:
            value = len(self.appear_vocabs) / self._total_vocabs
        if reset:
            self.reset()
        return value