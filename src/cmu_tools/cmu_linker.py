"""
    cmu_linker.py
    Created by pierre
    10/6/22 - 1:09 PM
    Description:
    Set of tools to link a GPT tokenizer to the CMU dictionary.
 """
from typing import Optional

import torch
from transformers import PreTrainedTokenizer

from src.cmu_tools.cmu_dictionary import CMUDictionary


class CMULinker:
    def __init__(self, tokenizer: PreTrainedTokenizer, cmu_dict: CMUDictionary):
        self.cmu_vocab: set[str] = set(cmu_dict.cmu_dictionary.keys())
        self.cmu_dictionary = cmu_dict
        self.gpt_vocab: dict[str, int] = tokenizer.get_vocab()
        self.pronunciation_mapping: list[Optional[str]] = [None] * len(self.gpt_vocab)  # Token to CMU
        self.token_mapping: dict[str, int] = {}  # CMU to token
        self.build_mappings()
        self.newline_id = self.gpt_vocab["Ċ"]

    def build_mappings(self) -> None:
        for word, token_id in self.gpt_vocab.items():
            if word[0] == 'Ġ':
                word = word[1:]
            word = word.lower()
            if word in self.cmu_vocab:
                self.pronunciation_mapping[token_id] = word
                self.token_mapping[word] = token_id

    def coverage(self) -> tuple[float, float]:
        ct_tokens = sum(w is not None for w in self.pronunciation_mapping)
        ct_word = len(self.token_mapping)
        print("Tokenizer coverage: {:.1%}% of the tokens found in CMU.".format(ct_tokens / len(self.gpt_vocab)))
        print("CMU coverage: {:.1%}% of the words in CMU found in the tokenizer.".format(ct_word / len(self.cmu_vocab)))
        return ct_tokens / len(self.gpt_vocab), ct_word / len(self.cmu_vocab)

    def get_not_found(self):
        res = []
        for word, token_id in self.gpt_vocab.items():
            if self.pronunciation_mapping[token_id] is None:
                res.append(word)
        return res

    def convert_ids(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, max_length=32):
        phonemes = []
        stress = []
        attention = []
        for token_id, mask in zip(torch.flatten(token_ids), torch.flatten(attention_mask)):
            if not mask:
                phonemes.append([-1] * max_length)
                stress.append([-1] * max_length)
                attention.append([0] * max_length)
                continue
            try:
                pho, s = self.cmu_dictionary.cmu_dictionary[self.pronunciation_mapping[token_id]][0]
            except KeyError:
                pho = s = []
            phonemes.append(pho + [-1] * (max_length - len(pho)) if len(pho) <= max_length else pho[:max_length])
            stress.append(s + [-1] * (max_length - len(s)) if len(s) <= max_length else s[:max_length])
            attention.append([1] * min(len(pho), max_length) + [0] * (max_length - min(len(pho), max_length)))

        phonemes = torch.reshape(
            torch.tensor(phonemes, device=token_ids.device, dtype=token_ids.dtype),
            (*token_ids.shape, max_length))
        stress = torch.reshape(
            torch.tensor(stress, device=token_ids.device, dtype=token_ids.dtype),
            (*token_ids.shape, max_length))
        attention = torch.reshape(
            torch.tensor(attention, device=token_ids.device, dtype=token_ids.dtype),
            (*token_ids.shape, max_length))
        return phonemes, stress, attention
