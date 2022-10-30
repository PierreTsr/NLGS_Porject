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
import multiprocessing as mp

from tqdm import tqdm

from .utils import to_nested_list
from src.cmu_tools import CMULinker, get_rhyming_part
from src.pronunciation_embeddings import PronunciationTokenizer


class RhymingMetrics:
    def __init__(self, cmu_linker: CMULinker, tokenizer: PronunciationTokenizer, workers: Optional[int] = None):
        self.linker = cmu_linker
        self.tokenizer = tokenizer
        self.workers = None

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

    def build_rhyme_df(self, generation: list[int], max_depth=6, pad: int = 0) -> pd.DataFrame:
        rhyming_words = self.get_rhyming_tokens(generation)
        df = {
            "stress": [],
            **{i: [] for i in range(max_depth)}
        }
        for word in rhyming_words:
            try:
                phonemes, stress = self.linker.get_pronunciation(word)
                rhyming_part = get_rhyming_part(phonemes, stress)
            except KeyError:
                rhyming_part = tuple(pd.NA for _ in range(max_depth + 1))
            finally:
                df["stress"].append(rhyming_part[0])
                for i in range(max_depth):
                    if i < len(rhyming_part) - 1:
                        df[i].append(rhyming_part[len(rhyming_part) - i - 1])
                    else:
                        df[i].append(-1)
        for key in df.keys():
            df[key] += [pd.NA] * pad
        df = pd.DataFrame.from_dict(df)
        df["rhyme"] = False
        return df

    def mark_rhymes(self, rhyme_df: pd.DataFrame):
        df = rhyme_df.dropna()
        max_depth = df.shape[1] - 2
        cols = list(range(max_depth))
        count = df.groupby(cols, as_index=False).size()
        rhymes = count[count["size"] > 1]
        rhymes = {(*row[1],) for row in rhymes[cols].iterrows()}
        mask = [(*row[1],) in rhymes for row in rhyme_df[cols].iterrows()]
        rhyme_df.loc[mask, "rhyme"] = True

    def count_rolling_rhymes(self, rhyme_df: pd.DataFrame, window: int) -> float:
        n = len(rhyme_df)
        for i in range(n - window + 1):
            self.mark_rhymes(rhyme_df.iloc[i:i + window, :])
        return len(rhyme_df[rhyme_df["rhyme"]])

    def _count_rolling_rhymes(self, args):
        return self.count_rolling_rhymes(*args)

    def avg_rolling_rhymes(self, generation: list[pd.DataFrame], window: int):
        r = 0
        n = sum(len(df) for df in generation)
        if self.workers is not None:
            args = [(df, window) for df in generation]
            n = len(args)
            with mp.Pool(self.workers) as pool:
                r = sum(tqdm(pool.imap_unordered(self._count_rolling_rhymes, args), total=n))
        else:
            for df in tqdm(generation):
                r += self.count_rolling_rhymes(df, window)
        return r / n

    def compute(self, generations: list[list[int]] | np.ndarray, max_depth: int = 4, window: int = 4):
        generations = to_nested_list(generations)
        rhymes_dfs = [self.build_rhyme_df(generation, max_depth) for generation in generations]
        metrics = {
            "rolling_rhymes": self.avg_rolling_rhymes(rhymes_dfs, window=window),
        }
        return metrics
