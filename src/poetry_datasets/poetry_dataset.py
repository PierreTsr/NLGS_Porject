"""
    poetry_dataset.py
    Created by Pierre Tessier
    10/17/22 - 9:06 PM
    Description:
    Tools to transform a pd.DataFrame corpus into the HuggingFace's Dataset format.
 """
from typing import Optional

from datasets import Dataset
from pandas import DataFrame
from transformers import PreTrainedTokenizer


def get_tokenizing_fn(tokenizer: PreTrainedTokenizer, **kwargs):
    def tokenizing_fn(data):
        return tokenizer(data["text"], return_tensors="pt", **kwargs)

    return tokenizing_fn


def get_pronunciation_tokenizing_fn(tokenizer: PreTrainedTokenizer, linker: Optional, max_length, **kwargs):
    def tokenizing_fn(data):
        tokens = tokenizer(data["text"], return_tensors="pt", **kwargs)
        tokens["pronunciation_ids"], tokens["stress_ids"], tokens["pronunciation_attention_mask"] = linker.convert_ids(
            tokens["input_ids"],
            tokens["attention_mask"],
            max_length=max_length if max_length is not None else 8
        )
        return tokens

    return tokenizing_fn


def build_dataset(corpus: DataFrame,
                  tokenizer: PreTrainedTokenizer,
                  linker=None,
                  max_length=8,
                  batched=True,
                  batch_size=1000,
                  num_proc=1,
                  **kwargs):

    dataset = Dataset.from_pandas(corpus[["text"]])
    if linker is not None:
        tokenizing_fn = get_pronunciation_tokenizing_fn(tokenizer, linker, max_length, **kwargs)
    else:
        tokenizing_fn = get_tokenizing_fn(tokenizer, **kwargs)
    print("Tokenizing dataset in PyTorch format:")
    dataset = dataset.with_format("torch")
    dataset = dataset.map(tokenizing_fn, batched=batched, batch_size=batch_size, num_proc=num_proc)
    return dataset
