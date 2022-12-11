"""
    meter.py
    Created by Pierre Tessier
    10/29/22 - 4:37 PM
    Description:
    # Enter file description
 """
import numpy as np
import torch
from tqdm import tqdm
from fastDamerauLevenshtein import damerauLevenshtein
from collections import  defaultdict

from src.cmu_tools import CMULinker
from src.pronunciation_embeddings import PronunciationTokenizer
from src.utils import to_nested_list


class MeterMetrics:
    mapping = ["", "x", "/", "/"]

    def __init__(self, cmu_linker: CMULinker, tokenizer: PronunciationTokenizer, patterns: list[str] = None,
                 verbose: bool = True):
        self.linker = cmu_linker
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.mapping_fn = lambda i: self.mapping[i]
        if patterns is None:
            self.patterns = [
                "x/x/x/x/x/",
                "/xx/x/x/x/",
                "x/x/x/x/x/x",
                "x/x/x/x/",
                "/x/x/x/x",
                "xx/xx/xx/",
                "/xx/xx/xx/xx/xx/x"
            ]
        else:
            self.patterns = patterns
        self.stress_to_idx = defaultdict(list)
        self.build_dicts()
        self.all_stresses = list(self.stress_to_idx.keys())

    def build_dicts(self):
        for i in range(len(self.linker.gpt_vocab)):
            try:
                pronunciations = self.linker.get_all_pronunciations(i)
            except KeyError:
                continue
            stress = set(["".join(map(self.mapping_fn, s)) for _, s in pronunciations])
            for s in stress:
                self.stress_to_idx[s].append(i)
        monosyllabic = list(set(self.stress_to_idx["x"] + self.stress_to_idx["/"]))
        self.stress_to_idx["x"] = monosyllabic
        self.stress_to_idx["/"] = monosyllabic
        self.stress_to_idx[""].append(self.tokenizer.newline_token)

    def all_damerauLevenshtein(self, line: list[int]):
        min_candidate = [""] * len(self.patterns)
        min_val = [1e9] * len(self.patterns)
        for word in line:
            try:
                l = self.linker.get_all_pronunciations(word)
            except KeyError:
                continue
            stress = set(["".join(map(self.mapping_fn, s)) for _, s in l])
            if "/" in stress or "x" in stress:
                stress.add("x")
                stress.add("/")
            for i, (candidate, p) in enumerate(zip(min_candidate, self.patterns)):
                m = 1e9
                res = candidate
                for s in stress:
                    d = damerauLevenshtein(candidate + s, p[:len(candidate) + len(s)], similarity=False)
                    if d < m:
                        m = d
                        res = candidate + s
                min_candidate[i] = res
                min_val[i] = m
        min_val_complete = [damerauLevenshtein(c, p, similarity=False) for c, p in zip(min_candidate, self.patterns)]
        return min_candidate, min_val, min_val_complete

    def min_damerauLevenshtein(self, line: list[int]) -> tuple[str, float, float]:
        min_candidate, min_val, min_val_complete = self.all_damerauLevenshtein(line)
        idx = min(range(len(min_val)), key=lambda i: min_val[i])
        return min_candidate[idx], min_val_complete[idx], min_val[idx]

    def damerauLevenshtein_all_vocab_score(self, line: list[int]):
        if self.tokenizer.newline_token in line:
            line = line[line.index(self.tokenizer.newline_token):]
        max_candidate, _, _ = self.all_damerauLevenshtein(line)
        max_val = torch.zeros((len(self.linker.gpt_vocab), len(self.all_stresses)))
        scores = [0] * len(self.all_stresses)
        for candidate, pattern in zip(max_candidate, self.patterns):
            for i, s in enumerate(self.all_stresses):
                n = len(candidate + s)
                scores[i] = max(scores[i], damerauLevenshtein(candidate + s, pattern))
        for i, s in enumerate(self.all_stresses):
            max_val[self.stress_to_idx[s], i] = scores[i]
        meter_scores = torch.max(max_val, dim=1).values
        return meter_scores * 2 - 1

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
            if "/" in stress:
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
            s += self.min_damerauLevenshtein(generation[start:stop])[1]
            n_lines += 1
        return s / n_lines

    def avg_meter(self, generations: list[list[int]]) -> float:
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
            "correct_meter_distance": self.avg_meter(generations)
        }
        return metrics
