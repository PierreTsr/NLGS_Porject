"""
    cmu_linker.py
    Created by pierre
    10/6/22 - 1:09 PM
    Description:
    Set of tools to link a GPT tokenizer to the CMU dictionary.
 """
from transformers import PreTrainedTokenizer
from typing import Optional

from cmu_dictionary import CMUDictionary


class CMULinker:
    def __init__(self, tokenizer: PreTrainedTokenizer, cmu_dict: CMUDictionary):
        self.cmu_vocab: set[str] = set(cmu_dict.cmu_dictionary.keys())
        self.gpt_vocab: dict[str, int] = tokenizer.get_vocab()
        self.pronunciation_mapping: list[Optional[str]] = [None] * len(self.gpt_vocab)  # Token to CMU
        self.token_mapping: dict[str, int] = {}  # CMU to token

    def build_mappings(self) -> None:
        for word, token_id in self.gpt_vocab.items():
            if word[0] == 'Ġ':
                word = word[1:]
            if word in self.cmu_vocab:
                self.pronunciation_mapping[token_id] = word
                self.token_mapping[word] = token_id


