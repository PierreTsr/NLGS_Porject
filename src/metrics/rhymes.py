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
    def __init__(self, cmu_linker: CMULinker, tokenizer: PronunciationTokenizer, window:int = 4, coeffs: tuple[int, ...] = (1/2,1/2), verbose: bool = True):
        self.linker = cmu_linker
        self.tokenizer = tokenizer
        self.window = window
        self.coeffs = coeffs
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

    def build_rhyme_list(self, generation: list[int]) -> list[tuple[int,tuple[int, ...]]]:
        rhyming_words = self.get_rhyming_tokens(generation)
        rhymes = []
        for word in rhyming_words:
            try:
                phonemes, stress = self.linker.get_pronunciation(word)
                rhyming_part = get_rhyming_part(phonemes, stress)[1:]
            except KeyError:
                rhyming_part = tuple()
            rhymes.append((word, rhyming_part))
        return rhymes

    def mark_rhymes(self, rhymes: list[tuple[int, tuple[int, ...]]], rhyming: list[bool], begin: int, end: int,
                    perfect: bool = True):
        for i in range(begin, end):
            for j in range(i+1, end):
                word_a, rhymes_a = rhymes[i]
                word_b, rhymes_b = rhymes[j]
                if word_a == word_b or (rhyming[i] and rhyming[j]):
                    continue
                if perfect:
                    b = bool(rhymes_a) and bool(rhymes_b) and (rhymes_a == rhymes_b)
                else:
                    b = bool(rhymes_a) and bool(rhymes_b) and (rhymes_a[0] == rhymes_b[0])
                rhyming[i] |= b
                rhyming[j] |= b

    def count_rhymes(self, rhymes: list[tuple[int,tuple[int, ...]]], window: int, perfect: bool = True) -> float:
        n = len(rhymes)
        rhyming = [False] * n
        for i in range(n):
            self.mark_rhymes(rhymes, rhyming, i, min(i + window, n), perfect)
        return sum(rhyming)

    def avg_rhymes(self, generation: list[list[tuple[int,tuple[int, ...]]]], window: int, perfect: bool = True):
        r = 0
        n = sum(len(rhymes) for rhymes in generation)
        if n == 0:
            return 0
        for rhymes in tqdm(generation, disable=not self.verbose, desc="Computing rhyme pairs"):
            r += self.count_rhymes(rhymes, window, perfect)
        return r / n

    def score_generation(self, generations: list[list[int]]):
        rhymes_lists = [self.build_rhyme_list(generation) for generation in
                        tqdm(generations, disable=not self.verbose, desc="Creating rhyming parts")]
        score = self.coeffs[0] * self.avg_rhymes(rhymes_lists, window=self.window, perfect=True) + \
            self.coeffs[1] * self.avg_rhymes(rhymes_lists, window=self.window, perfect=False)
        return score

    def compute(self, generations: list[list[int]] | np.ndarray):
        generations = to_nested_list(generations)
        rhymes_lists = [self.build_rhyme_list(generation) for generation in
                        tqdm(generations, disable=not self.verbose, desc="Creating rhyming parts")]
        metrics = {
            "perfect_rhymes": self.avg_rhymes(rhymes_lists, window=self.window, perfect=True),
            "weak_rhymes": self.avg_rhymes(rhymes_lists, window=self.window, perfect=False),
        }
        return metrics
