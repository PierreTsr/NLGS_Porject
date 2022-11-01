"""
    rhymes.py
    Created by Pierre Tessier
    10/27/22 - 3:04 PM
    Description:
    # Enter file description
 """
from typing import Optional

import numpy as np
import pandas as pd

from tqdm import tqdm

from .utils import to_nested_list
from src.cmu_tools import CMULinker, get_rhyming_part
from src.pronunciation_embeddings import PronunciationTokenizer


class RhymingMetrics:
    def __init__(self, cmu_linker: CMULinker, tokenizer: PronunciationTokenizer, verbose: bool = True):
        self.linker = cmu_linker
        self.tokenizer = tokenizer
        self.verbose = verbose

    def get_rhyming_tokens(self, generation: list[int]) -> list[int]:
        newlines = [i for i, x in enumerate(generation) if x == self.tokenizer.newline_token]
        rhyming_tokens = []
        last = 0
        for i in newlines:
            idx = i
            while last < idx:
                if generation[idx] in self.tokenizer.punctuation_ids.keys():
                    idx -= 1
                else:
                    rhyming_tokens.append(generation[idx])
                    break
        return rhyming_tokens

    def build_rhyme_list(self, generation: list[int]) -> list[tuple[int, ...]]:
        rhyming_words = self.get_rhyming_tokens(generation)
        rhymes = []
        for word in rhyming_words:
            try:
                phonemes, stress = self.linker.get_pronunciation(word)
                rhyming_part = get_rhyming_part(phonemes, stress)[1:]
            except KeyError:
                rhyming_part = tuple()
            rhymes.append(rhyming_part)
        return rhymes

    def mark_perfect_rhymes(self, rhymes: list[tuple[int, ...]], rhyming: list[bool], begin: int, end: int):
        for i in range(begin, end):
            for j in range(begin, end):
                if i == j or (rhyming[i] and rhyming[j]):
                    continue
                b = rhymes[i] == rhymes[j]
                rhyming[i] |= b
                rhyming[j] |= b

    def count_rolling_rhymes(self, rhymes: list[tuple[int, ...]], window: int) -> float:
        n = len(rhymes)
        rhyming = [False] * n
        for i in range(n - window + 1):
            self.mark_perfect_rhymes(rhymes, rhyming, i, i+window)
        return sum(rhyming)

    def avg_rolling_rhymes(self, generation: list[list[tuple[int, ...]]], window: int):
        r = 0
        n = sum(len(rhymes) for rhymes in generation)
        for df in tqdm(generation, disable=not self.verbose, desc="Computing rhyme pairs"):
            r += self.count_rolling_rhymes(df, window)
        return r / n

    def compute(self, generations: list[list[int]] | np.ndarray, window: int = 4):
        generations = to_nested_list(generations)
        rhymes_lists = [self.build_rhyme_list(generation) for generation in tqdm(generations, disable=not self.verbose, desc="Creating rhyming parts")]
        metrics = {
            "rolling_rhymes": self.avg_rolling_rhymes(rhymes_lists, window=window),
        }
        return metrics
