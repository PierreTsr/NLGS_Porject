"""
    gutenberg_poetry_corpus.py
    Created by Pierre Tessier
    10/17/22 - 7:01 PM
    Description:
    Load the Gutenberg Poetry Corpus
 """
import json
from pathlib import Path

import pandas as pd


def build_corpus(path: Path = Path("data/gutenberg_poetry/gutenberg_poetry_corpus")):
    with open(path, "r") as file:
        l = []
        for line in file:
            l.append(json.loads(line.strip()))
    corpus = {}
    for d in l:
        line, gid = d["s"], int(d["gid"])
        if gid not in corpus.keys():
            corpus[gid] = []
        corpus[gid].append(line)
    corpus = {gid: "\n".join(lines) for gid, lines in corpus.items()}
    corpus = pd.DataFrame.from_dict(corpus, orient="index", columns=["text"])
    return corpus


def add_metadata(corpus: pd.DataFrame, path: Path = Path("data/gutenberg_poetry/gutenberg_metadata.json")):
    metadata = pd.read_json(path)
    corpus = pd.merge(corpus, metadata, how="left", left_index=True, right_on="gd-num-padded").reset_index(drop=True)
    return corpus


def load_gutenberg(path: Path = Path("data/gutenberg_poetry")):
    corpus = build_corpus(path / "gutenberg_poetry_corpus")
    corpus = add_metadata(corpus, path / "gutenberg_metadata.json")
    return corpus
