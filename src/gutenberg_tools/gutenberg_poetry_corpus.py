"""
    gutenberg_poetry_corpus.py
    Created by Pierre Tessier
    10/17/22 - 7:01 PM
    Description:
    Load the Gutenberg Poetry Corpus
 """
import json
from pathlib import Path
from random import shuffle
from typing import Union

import pandas as pd
from tqdm import tqdm


def build_corpus(path: Path = Path("data/gutenberg_poetry/gutenberg_poetry_corpus")) -> dict[int, list[str]]:
    print("Loading the Gutenberg Poetry Corpus from disk:")
    with open(path, "r") as file:
        l = []
        for line in tqdm(file, total=3085117):
            l.append(json.loads(line.strip()))
    corpus = {}
    for d in l:
        line, gid = d["s"], int(d["gid"])
        if gid not in corpus.keys():
            corpus[gid] = []
        corpus[gid].append(line)
    return corpus


def add_metadata(corpus: pd.DataFrame,
                 path: Path = Path("data/gutenberg_poetry/gutenberg_metadata.json")) -> pd.DataFrame:
    metadata = pd.read_json(path)
    corpus = pd.merge(corpus, metadata, how="left", left_on="gid", right_on="gd-num-padded").reset_index(drop=True)
    return corpus


def split(corpus: dict[int, list[str]], n_lines: int = 32, stride: int = 16) -> pd.DataFrame:
    rows = []
    for gid, lines in corpus.items():
        for i in range(0, len(lines), stride):
            end = min(i + n_lines, len(lines))
            rows.append({
                "gid": gid,
                "line": i,
                "text": "\n".join(lines[i:end]) + "\n"
            })
    return pd.DataFrame.from_dict(rows)


def load_gutenberg(path: Path = Path("data/gutenberg_poetry"), **kwargs):
    corpus = build_corpus(path / "gutenberg_poetry_corpus")
    if kwargs["n_lines"] is not None:
        corpus = split(corpus, kwargs["n_lines"], kwargs["stride"])
    else:
        corpus = {gid: "\n".join(lines) + "\n" for gid, lines in corpus.items()}
        corpus = pd.DataFrame.from_dict(corpus, orient="index", columns=["text"])
        corpus = corpus.reset_index(names=["gid"])
        corpus["line"] = 0
    corpus = add_metadata(corpus, path / "gutenberg_metadata.json")
    return corpus


def load_split(path: Union[str, Path], data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    with open(path, "r") as file:
        split_dict = json.load(file)
    return {
        key: data.loc[data["gid"].isin(val)].copy() for key, val in split_dict.items()
    }


def split_gutenberg_to_json(
        path: Union[str, Path],
        data: pd.DataFrame,
        ratios: tuple[int, int, int] = (0.9, .05, .05),
) -> None:
    assert abs(sum(ratios) - 1) < 1e-5
    train = []
    test = []
    train_size = test_size = 0
    n = len(data)
    group_ids = list(set(data["gid"]))
    shuffle(group_ids)
    while train_size < ratios[0] * n:
        gid = group_ids.pop()
        train_size += len(data.loc[data["gid"] == gid].index)
        train.append(gid)
    while test_size + train_size < (ratios[0] + ratios[1]) * n:
        gid = group_ids.pop()
        test_size += len(data.loc[data["gid"] == gid].index)
        test.append(gid)
    validation = group_ids
    split_dict = {"train": train, "test": test, "validation": validation}
    with open(path, "w") as file:
        json.dump(split_dict, file)
