"""
    pronunciation_embeddings.py
    Created by Pierre Tessier
    10/19/22 - 10:06 AM
    Description:
    Train word2vec embeddings for pronunciation and stress tokens.
 """
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from src.pronunciation_embeddings.pronunciation_tokenizer import PronunciationTokenizer


def save_embeddings(model: Word2Vec, target_file: Path, voc_size: int):
    embeddings = []
    n = model.vector_size
    for i in range(voc_size):
        try:
            x = np.copy(model.wv[str(i)])
            embeddings.append(torch.Tensor(x))
        except KeyError:
            embeddings.append(torch.zeros(n))
    embeddings = torch.stack(embeddings)
    torch.save(embeddings, target_file)


def train(tokenizer: PronunciationTokenizer, dataset_dir: Path = Path("data/gutenberg_poetry_pronunciation"),
          target_dir: Path = Path("etc/pronunciation_embeddings/"), model_s={}, model_p={}, **kwargs):
    target_dir.mkdir(exist_ok=True, parents=True)

    model_p = Word2Vec(sentences=LineSentence(dataset_dir / "pronunciation.txt"), **model_p, **kwargs)
    save_embeddings(model_p, target_dir / "pronunciation_embeddings.pt", len(tokenizer.vocabulary_p))

    model_s = Word2Vec(sentences=LineSentence(dataset_dir / "stress.txt"), **model_s, **kwargs)
    save_embeddings(model_s, target_dir / "stress_embeddings.pt", len(tokenizer.vocabulary_s))
