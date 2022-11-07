"""
    distinct.py
    Created by Pierre Tessier
    11/7/22 - 5:17 PM
    Description:
    # Enter file description
 """
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.metrics.utils import to_nested_list


class DistinctMetrics:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_n: int = 4, verbose: bool = True):
        self.bos_token = tokenizer.bos_token_id
        self.eos_token = tokenizer.eos_token_id
        self.max_n = max_n
        self.verbose = verbose

    def compute_ngrams(self, text: list[int], n: int):
        res = []
        text = [self.bos_token] * (n - 1) + text + [self.eos_token] * (n - 1)
        for i in range(len(text) - n + 1):
            res.append(tuple(text[i:i + n]))
        return res

    def compute_distinct(self, text: list[int], n: int):
        ngrams = self.compute_ngrams(text, n)
        distinct = set(ngrams)
        return len(distinct) / len(ngrams)

    def avg_distinct(self, generations: list[list[int]], n: int):
        tot = len(generations)
        s = sum(self.compute_distinct(text, n) for text in
                tqdm(generations, disable=not self.verbose, desc="Computing distinct-{}".format(n)))
        return s / tot

    def compute(self, generations: list[list[int]]):
        generations = to_nested_list(generations)
        metrics = {
            "distinct-{}".format(n): self.avg_distinct(generations, n) for n in range(1, self.max_n + 1)
        }
        return metrics
