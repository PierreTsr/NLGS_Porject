"""
    rhymes.py
    Created by Pierre Tessier
    10/27/22 - 3:04 PM
    Description:
    # Enter file description
 """

import numpy as np
from tqdm import tqdm

from src.cmu_tools import CMULinker, get_rhyming_part
from src.pronunciation_embeddings import PronunciationTokenizer
from src.utils import to_nested_list


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

    def mark_rhymes(self, rhymes: list[tuple[int, ...]], rhyming: list[bool], begin: int, end: int,
                    perfect: bool = True):
        for i in range(begin, end):
            for j in range(begin, end):
                if i == j or (rhyming[i] and rhyming[j]):
                    continue
                if perfect:
                    b = bool(rhymes[i]) and bool(rhymes[j]) and (rhymes[i] == rhymes[j])
                else:
                    b = bool(rhymes[i]) and bool(rhymes[j]) and (rhymes[i][0] == rhymes[j][0])
                rhyming[i] |= b
                rhyming[j] |= b

    def count_rhymes(self, rhymes: list[tuple[int, ...]], window: int, perfect: bool = True) -> float:
        n = len(rhymes)
        rhyming = [False] * n
        for i in range(n - window + 1):
            self.mark_rhymes(rhymes, rhyming, i, i + window, perfect)
        return sum(rhyming)

    def avg_rhymes(self, generation: list[list[tuple[int, ...]]], window: int, perfect: bool = True):
        r = 0
        n = sum(len(rhymes) for rhymes in generation)
        if n == 0:
            return 0
        for rhymes in tqdm(generation, disable=not self.verbose, desc="Computing rhyme pairs"):
            r += self.count_rhymes(rhymes, window, perfect)
        return r / n

    def compute(self, generations: list[list[int]] | np.ndarray, window: int = 4):
        generations = to_nested_list(generations)
        rhymes_lists = [self.build_rhyme_list(generation) for generation in
                        tqdm(generations, disable=not self.verbose, desc="Creating rhyming parts")]
        metrics = {
            "perfect_rhymes": self.avg_rhymes(rhymes_lists, window=window, perfect=True),
            "weak_rhymes": self.avg_rhymes(rhymes_lists, window=window, perfect=False),
        }
        return metrics
