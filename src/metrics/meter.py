"""
    meter.py
    Created by Pierre Tessier
    10/29/22 - 4:37 PM
    Description:
    # Enter file description
 """
import numpy as np
from tqdm import tqdm

from src.cmu_tools import CMULinker
from src.metrics.utils import to_nested_list
from src.pronunciation_embeddings import PronunciationTokenizer


class MeterMetrics:
    mapping = ["", "x", "/", "/"]

    def __init__(self, cmu_linker: CMULinker, tokenizer: PronunciationTokenizer, patterns: set[str] = None, verbose:bool=True):
        self.linker = cmu_linker
        self.tokenizer = tokenizer
        self.verbose=verbose
        self.mapping_fn = lambda i: self.mapping[i]
        if patterns is None:
            self.patterns = {
                "x/x/x/x/x/",
                "/xx/x/x/x/",
                "x/x/x/x/x/x",
                "x/x/x/x/",
                "/x/x/x/x",
                "xx/xx/xx/",
                "/xx/xx/xx/xx/xx/x"
            }
        else:
            self.patterns = patterns

    def get_line_stress(self, line: list[int]) -> set[str]:
        stress_line = [""]
        for word in line:
            try:
                l = self.linker.get_all_pronunciations(word)
            except KeyError:
                continue
            stress = set(["".join(map(self.mapping_fn, s)) for _, s in l])
            # if"/" in stress:
            #     stress.add("x")
            new_stress_line = []
            for sl in stress_line:
                for s in stress:
                    new_stress_line.append(sl + s)
            stress_line = new_stress_line
        return set(stress_line)

    def check_meter_line(self, line: list[int]):
        stress_line = [""]
        for word in line:
            try:
                l = self.linker.get_all_pronunciations(word)
            except KeyError:
                continue
            stress = set(["".join(map(self.mapping_fn, s)) for _, s in l])
            if"/" in stress:
                 stress.add("x")
            new_stress_line = []
            for sl in stress_line:
                for s in stress:
                    candidate = sl + s
                    if any([candidate == p[:len(candidate)] for p in self.patterns]):
                        new_stress_line.append(candidate)
            stress_line = new_stress_line
            if not stress_line:
                return False
        return bool(set(stress_line) & self.patterns)

    def count_meter_rate(self, generation: list[int]):
        newlines = [0] + [i for i, x in enumerate(generation) if x == self.tokenizer.newline_token] + [len(generation)]
        s = 0
        n_lines = 0
        for i in range(len(newlines) - 1):
            start, stop = newlines[i:i + 2]
            if stop - start <= 1:
                continue
            s += self.check_meter_line(generation[start:stop])
            n_lines += 1
        return s / n_lines

    def avg_meter(self, generations: list[list[int]]):
        n = len(generations)
        if n == 0:
            return 0
        s = 0
        for generation in tqdm(generations, disable=not self.verbose, desc="Computing meter matches"):
            s += self.count_meter_rate(generation)
        return s / n

    def compute(self, generations: list[int] | np.ndarray):
        generations = to_nested_list(generations)
        metrics = {
            "correct_meter_frequency": self.avg_meter(generations)
        }
        return metrics
