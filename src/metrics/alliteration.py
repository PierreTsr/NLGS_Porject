"""
    alliteration.py
    Created by Pierre Tessier
    10/27/22 - 3:04 PM
    Description:
    # Enter file description
 """
import numpy as np
from tqdm import tqdm

from src.cmu_tools import CMULinker
from src.pronunciation_embeddings import PronunciationTokenizer
from .utils import to_nested_list


class AlliterationMetrics:
    def __init__(self, cmu_linker: CMULinker,
                 tokenizer: PronunciationTokenizer,
                 thresholds: tuple[int, ...] = (2, 3, 4),
                 verbose: bool = True):
        self.linker = cmu_linker
        self.tokenizer = tokenizer
        self.threshold = thresholds
        self.verbose = verbose

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
        newlines = [0] + [i for i, x in enumerate(generation) if x == self.tokenizer.newline_token] + [len(generation)]
        consonants = []
        for i in range(len(newlines) - 1):
            start, stop = newlines[i:i + 2]
            if stop - start <= 1:
                continue
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
        for accent in tqdm(accentuated, disable=not self.verbose, desc="Computing alliterations-{}".format(threshold)):
            a += self.count_alliterations(accent, threshold)
            n += len(accent)
        return a / n

    def compute(self, generations: list[int] | np.ndarray):
        generations = to_nested_list(generations)
        accentuated_cons = [self.get_accentuated_consonants(generation) for generation in
                            tqdm(generations, disable=not self.verbose, desc="Creating phonemes sequences")]
        metrics = {
            "alliteration_{}".format(t): self.avg_alliteration(accentuated_cons, t) for t in self.threshold
        }
        return metrics
