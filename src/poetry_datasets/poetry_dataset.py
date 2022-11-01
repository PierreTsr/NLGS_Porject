"""
    poetry_dataset.py
    Created by Pierre Tessier
    10/17/22 - 9:06 PM
    Description:
    Tools to transform a pd.DataFrame corpus into the HuggingFace's Dataset format.
 """
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Array2D, Value, load_from_disk
from pandas import DataFrame
from transformers import PreTrainedTokenizer

from src.pronunciation_embeddings import PronunciationTokenizer

type_mappings = {
    "input_ids": torch.long,
    "attention_mask": torch.bool,
    "pronunciation": torch.uint8,
    "stress": torch.uint8,
    "pronunciation_attention_mask": torch.bool
}


def get_tokenizing_fn(tokenizer: PreTrainedTokenizer, **kwargs):
    def tokenizing_fn(data):
        data = tokenizer(data["text"], **kwargs)
        return data | {"labels": data["input_ids"]}

    return tokenizing_fn


def get_pronunciation_tokenizing_fn(tokenizer: PreTrainedTokenizer, tokenizer_p: PronunciationTokenizer, max_length,
                                    **kwargs):
    def tokenizing_fn(data):
        tokens = tokenizer(data["text"], **kwargs)
        pronunciation = tokenizer_p.convert_tokens_3d(
            tokens["input_ids"],
            tokens["attention_mask"],
            max_length=max_length if max_length is not None else 8
        )
        tokens = tokens | pronunciation | {"labels": tokens["input_ids"]}
        return tokens

    return tokenizing_fn


def custom_getter(chunk):
    data = {}
    for key, val in chunk.items():
        if key in type_mappings.keys():
            data[key] = torch.tensor(val, dtype=type_mappings[key])
    return data


def load_dataset(path: Path | str):
    dataset = load_from_disk(str(path))
    dataset.set_transform(custom_getter)
    return dataset


def build_dataset(corpus: DataFrame | dict[str, DataFrame],
                  tokenizer: PreTrainedTokenizer,
                  tokenizer_p: Optional[PronunciationTokenizer] = None,
                  max_length=8,
                  batched=True,
                  batch_size=1000,
                  num_proc=1,
                  **kwargs):
    if tokenizer_p is not None:
        features = Features({
            "text": Value("string"),
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("uint8")),
            "pronunciation": Array2D((-1, max_length), "uint8"),
            "stress": Array2D((-1, max_length), "uint8"),
            "pronunciation_attention_mask": Array2D((-1, max_length), "uint8"),
            "labels": Sequence(Value("int32")),
        })
        tokenizing_fn = get_pronunciation_tokenizing_fn(tokenizer, tokenizer_p, max_length, **kwargs)
    else:
        features = Features({
            "text": Value("string"),
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("uint8")),
            "labels": Sequence(Value("int32")),
        })
        tokenizing_fn = get_tokenizing_fn(tokenizer, **kwargs)
    if type(corpus) == dict:
        dataset = DatasetDict({
            key: Dataset.from_pandas(df[["text"]], preserve_index=False) for key, df in corpus.items()
        })
    else:
        dataset = Dataset.from_pandas(corpus[["text"]], preserve_index=False)
    print("Tokenizing dataset in PyTorch format:")
    dataset = dataset.map(tokenizing_fn, batched=batched, batch_size=batch_size, num_proc=num_proc, features=features)
    return dataset
