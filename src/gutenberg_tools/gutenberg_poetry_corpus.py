"""
    gutenberg_poetry_corpus.py
    Created by Pierre Tessier
    10/17/22 - 7:01 PM
    Description:
    Load the Gutenberg Poetry Corpus
 """
import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd


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
    corpus = pd.merge(corpus, metadata, how="left", left_index=True, right_on="gd-num-padded").reset_index(drop=True)
    return corpus


def split(corpus: dict[int, list[str]], n_lines: int = 32, stride: int=16) -> pd.DataFrame:
    rows = []
    for gid, lines in corpus.items():
        for i in range(0,len(lines), stride):
            end = min(i+n_lines, len(lines))
            rows.append({
                "gid": gid,
                "line": i,
                "text": "\n".join(lines[i:end])
            })
    return pd.DataFrame.from_dict(rows)


def load_gutenberg(path: Path = Path("data/gutenberg_poetry"), **kwargs):
    corpus = build_corpus(path / "gutenberg_poetry_corpus")
    if "n_lines" in kwargs.keys() and "stride" in kwargs.keys():
        corpus = split(corpus, kwargs["n_lines"], kwargs["stride"])
    else:
        corpus = {gid: "\n".join(lines) for gid, lines in corpus.items()}
        corpus = pd.DataFrame.from_dict(corpus, orient="index", columns=["text"])
        corpus = corpus.reset_index(names=["gid"])
        corpus["line"] = 0
    corpus = add_metadata(corpus, path / "gutenberg_metadata.json")
    return corpus
