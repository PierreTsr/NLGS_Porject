"""
    pronunciation_gpt.py
    Created by Pierre Tessier
    10/20/22 - 4:54 PM
    Description:
    # Enter file description
 """
import os
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPTNeoForCausalLM, TrainingArguments, TrainerState, TrainerControl, \
    DefaultFlowCallback, GPTNeoConfig

from . import PronunciationAttention


class PronunciationGPT(GPTNeoForCausalLM):

    def __init__(self, gpt: GPTNeoForCausalLM, embeddings_p: torch.Tensor, embeddings_s: torch.Tensor,
                 **kwargs):
        # super().__init__(gpt.config)
        for attr, val in vars(gpt).items():
            vars(self)[attr] = val

        self.pronunciation = PronunciationAttention(
            embeddings_p,
            embeddings_s,
            self.get_input_embeddings().embedding_dim,
            **kwargs
        )
        self.y = torch.nn.Parameter(torch.tensor(1e-1), requires_grad=True)
        self.switch_pronunciation = True

    def enable_pronunciation(self):
        self.switch_pronunciation = True

    def disable_pronunciation(self):
        self.switch_pronunciation = False

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        return super().resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        return super().get_position_embeddings()

    def freeze_gpt(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def unfreeze_gpt(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            pronunciation: Optional[torch.Tensor] = None,
            stress: Optional[torch.Tensor] = None,
            pronunciation_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        base_embeddings = self.get_input_embeddings()(input_ids)

        if pronunciation is None or stress is None or pronunciation_attention_mask is None:
            raise ValueError("PronunciationGPT needs pronunciation, stress and pronunciation_attention_mask inputs.")

        if self.switch_pronunciation:
            pronunciation_embeddings = self.pronunciation(
                pronunciation,
                stress,
                pronunciation_attention_mask
            )
            inputs_embeds = base_embeddings + self.y * pronunciation_embeddings
        else:
            inputs_embeds = base_embeddings

        outputs = super().forward(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class MixinValueCallback(DefaultFlowCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               logs: dict[str, float] = None, model: PronunciationGPT = None, **kwargs):
        y = model.y.data.item()
        logs["mixin_value"] = y
