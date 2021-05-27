#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn.modules.rnn import LSTMCell
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from CY_DataReadandMetric import *
from overrides import overrides
from allennlp.data.fields import Field, TextField, MetadataField, MultiLabelField, ListField
import torch
from allennlp.training.metrics import Average
import pkuseg
from allennlp.nn.util import get_text_field_mask


@Model.register("simple_seq2seq1")
class SimpleSeq2Seq(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        context_encoder: Seq2VecEncoder,
        max_decoding_steps: int = 32,
        attention: Attention = None,
        target_namespace: str = "tokens",
        scheduled_sampling_ratio: float = 0.0,
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio  # Maybe we can try
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self.pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)
        # self.outfeature = 600
        self._max_decoding_steps = max_decoding_steps
        self.kd_metric = KD_Metric()
        self.bleu_aver = NLTK_BLEU(ngram_weights=(0.25, 0.25, 0.25, 0.25))
        self.bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0))
        self.bleu2 = NLTK_BLEU(ngram_weights=(0, 1, 0, 0))
        self.bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1))
        self.dink1 = Distinct1()
        self.dink2 = Distinct2()
        self.topic_acc = Average()
        # anything about module
        self._source_embedder = source_embedder
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        target_embedding_dim = source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        self._encoder = encoder
        self.context_encoder = context_encoder
        self._encoder_output_dim = self._encoder.get_output_dim() # 512  要不把前两个都换成outfeater得了
        self._decoder_output_dim = self._encoder_output_dim
        self._decoder_input_dim = target_embedding_dim + self._decoder_output_dim
        self._attention = None
        # if attention:
        #     self._attention = attention
        #     self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim

        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._output_projection_layer = Linear(self._encoder_output_dim, num_classes)
        self.clac_num = 0

    @overrides
    def forward(self, next_sym, source_tokens, target_tokens, **args):
        self.clac_num += 1
        embedded_input = self._source_embedder(args['history'])  # bs * sen_num * seq_len * embedding
        source_mask = util.get_text_field_mask(args['history'], num_wrapping_dims=1)
        sentence_mask = util.get_text_field_mask(args['history'])
        sz = embedded_input.size()
        bs = source_mask.size(0)
        encoder_outputs = self._encoder(embedded_input.view(sz[0] * sz[1], sz[2], sz[3]), source_mask.view(sz[0] * sz[1], sz[2]))
        encoder_outputs = encoder_outputs.view(sz[0], sz[1], -1)
        contenxt_hidden = self.context_encoder(encoder_outputs, sentence_mask)
        # final_encoder_output = util.get_final_encoder_states(encoder_outputs, source_mask, self._encoder.is_bidirectional())
        state = {
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs,
            "decoder_hidden": encoder_outputs[:, -1, :].squeeze(1),
            "decoder_context": encoder_outputs.new_zeros(bs, self._decoder_output_dim),
            "context_hidden": contenxt_hidden
        }

        # 获取一次decoder
        output_dict = self._forward_loop(state, target_tokens)
        best_predictions = output_dict["predictions"]

        # output something
        references, hypothesis = [], []
        for i in range(bs):
            cut_hypo = best_predictions[i][:]
            if self._end_index in list(best_predictions[i]):
                cut_hypo = best_predictions[i][:list(best_predictions[i]).index(self._end_index)]
            hypothesis.append([self.vocab.get_token_from_index(idx.item()) for idx in cut_hypo])
        flag = 1
        for i in range(bs):
            cut_ref = target_tokens['tokens'][1:]
            if self._end_index in list(target_tokens['tokens'][i]):
                cut_ref = target_tokens['tokens'][i][1:list(target_tokens['tokens'][i]).index(self._end_index)]
            references.append([self.vocab.get_token_from_index(idx.item()) for idx in cut_ref])

        self.bleu_aver(references, hypothesis)
        self.bleu1(references, hypothesis)
        self.bleu2(references, hypothesis)
        self.bleu4(references, hypothesis)
        self.kd_metric(references, hypothesis)
        self.dink1(hypothesis)
        self.dink2(hypothesis)
        # output_dict['loss'] = output_dict['loss'] + 2 * topic_loss
        return output_dict

    def _forward_loop(
        self, state: Dict[str, torch.Tensor],target_tokens: Dict[str, torch.LongTensor] = None, his_sym: torch.Tensor=None
    ) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]
        num_decoding_steps = self._max_decoding_steps
        # print("yes?")
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]
            _, target_sequence_length = targets.size()
            if self.training:
                num_decoding_steps = target_sequence_length - 1

        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)  # (bs,)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training:
                input_choices = targets[:, timestep]
            else:
                input_choices = last_predictions
            #获取一次的decoder结果
            output_projections, state = self._prepare_output_projections(input_choices, state,his_sym)  # bs * num_class
            step_logits.append(output_projections.unsqueeze(1))
            class_probabilities = F.softmax(output_projections, dim=-1)  # bs * num_class
            _, predicted_classes = torch.max(class_probabilities, 1)  # (bs,)

            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))

        predictions = torch.cat(step_predictions, 1)  # bs * decoding_step

        output_dict = {"predictions": predictions}

        if self.training:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)
            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _prepare_output_projections(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor],
                                    his_sym: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        encoder_outputs = state["encoder_outputs"]  # bs, seq_len, encoder_output_dim
        source_mask = state["source_mask"]  # bs * seq_len
        decoder_hidden = state["decoder_hidden"]  # bs, decoder_output_dim
        decoder_context = state["decoder_context"]  # bs * decoder_output

        embedded_input = self._target_embedder(last_predictions)  # bs * target_embedding
        decoder_input = torch.cat((embedded_input, state['context_hidden']), -1)
        if self._attention:  # 如果加了seq_to_seq attention
            input_weights = self._attention(decoder_hidden, encoder_outputs, source_mask.float())  # bs * seq_len
            attended_input = util.weighted_sum(encoder_outputs, input_weights)  # bs * encoder_output
            decoder_input = torch.cat((attended_input, embedded_input), -1)  # bs * (decoder_output + target_embedding)

        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        state["decoder_hidden"] = decoder_hidden  # bs * hidden
        state["decoder_context"] = decoder_context

        # output_projections = self._output_projection_layer(torch.cat((decoder_hidden,graph_hidden),-1))
        output_projections = self._output_projection_layer(decoder_hidden)
        # sz = output_projections.size(0)
        # for b in range(sz):
        #     for k,li in enumerate(self.idx_to_vocab_list):
        #         if his_sym[b][k].item() == 1:
        #             output_projections[b][li] = 1e-9
        return output_projections, state

    @staticmethod
    def _get_loss(logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.LongTensor) -> torch.Tensor:

        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()  # bs * decoding_step

        return util.sequence_cross_entropy_with_logits(logits.contiguous(), relevant_targets, relevant_mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics.update(self.kd_metric.get_metric(reset=reset))
        all_metrics.update({"BLEU_avg": self.bleu_aver.get_metric(reset=reset)})
        all_metrics.update({"BLEU1": self.bleu1.get_metric(reset=reset)})
        all_metrics.update({"dink1": self.dink1.get_metric(reset=reset)})
        all_metrics.update({"dink2": self.dink2.get_metric(reset=reset)})
        # all_metrics.update({"topic_acc": self.topic_acc.get_metric(reset=reset)})
        return all_metrics




