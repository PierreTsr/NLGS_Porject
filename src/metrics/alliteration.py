"""
    alliteration.py
    Created by Pierre Tessier
    10/27/22 - 3:04 PM
    Description:
    # Enter file description
 """
import numpy as np

from .utils import to_nested_list
from src.cmu_tools import CMULinker
from src.pronunciation_embeddings import PronunciationTokenizer


class AlliterationMetrics:
    def __init__(self, cmu_linker: CMULinker, tokenizer: PronunciationTokenizer):
        self.linker = cmu_linker
        self.tokenizer = tokenizer

    def get_accentuated_consonants_line(self, line: list[int]) -> list[int]:
        consonants = []
        for word in line:
            try:
                phonemes, stress = self.linker.get_pronunciation(word)
            except KeyError:
                continue
            if phonemes[0] in self.linker.cmu_dictionary.consonants:
                consonants.append(phonemes[0])
            for idx, s in enumerate(stress):
                if idx < 2:
                    continue
                if s > 0 and phonemes[idx - 1] in self.linker.cmu_dictionary.consonants:
                    consonants.append(phonemes[idx - 1])
        return consonants

    def get_accentuated_consonants(self, generation: list[int]) -> list[list[int]]:
        newlines = [0] + [i for i, x in enumerate(generation) if x == self.tokenizer.newline_token]
        consonants = []
        for i in range(len(newlines) - 1):
            start, stop = newlines[i:i + 2]
            consonants.append(self.get_accentuated_consonants_line(generation[start + 1:stop]))
        return consonants

    def count_alliterations(self, accentuated: list[list[int]], threshold: int) -> int:
        n = 0
        for cons in accentuated:
            counts = {}
            for c in cons:
                if c not in counts.keys():
                    counts[c] = 0
                counts[c] += 1
            for val in counts.values():
                if val > threshold:
                    n += 1
        return n

    def avg_alliteration(self, accentuated: list[list[list[int]]], threshold: int):
        a = 0
        n = 0
        for accent in accentuated:
            a += self.count_alliterations(accent, threshold)
            n += len(accent)
        return a / n

    def compute(self, generations: list[int] | np.ndarray, threshold: int = 2):
        generations = to_nested_list(generations)
        accentuated_cons = [self.get_accentuated_consonants(generation) for generation in generations]
        metrics = {
            "alliteration_per_line": self.avg_alliteration(accentuated_cons, threshold)
        }
        return metrics
