"""
    evaluate_model.py
    Created by Pierre Tessier
    10/30/22 - 1:00 PM
    Description:
    # Enter file description
 """
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM

from src import load_dataset, CMUDictionary, CMULinker, PronunciationTokenizer
from src.metrics import RhymingMetrics, AlliterationMetrics, MeterMetrics


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: str = field(
        metadata={
            "help": "Path to the dataset dictionary for training, testing and evaluation."
        }
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Number of generations to process at once."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_inputs: list[str] = field(
        metadata={"help": "Name of the fields to query in the dataset."}
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Turn off CUDA during inference"}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."}
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={"help": "If set to int > 0, all ngrams of that size can only occur once."}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that"
                          " add up to top_p or higher are kept for generation."}
    )
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )


def main(model_args: ModelArguments, data_args: DataTrainingArguments):
    dataset = load_dataset(data_args.dataset_path)["validation"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    device = torch.device("cpu")
    if not model_args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    metrics = [
        RhymingMetrics(linker, tokenizer_p, verbose=False),
        AlliterationMetrics(linker, tokenizer_p, verbose=False),
        MeterMetrics(linker, tokenizer_p, verbose=False),
    ]

    sampler = BatchSampler(RandomSampler(dataset), batch_size=data_args.batch_size, drop_last=False)
    validation_loader = DataLoader(dataset, sampler=sampler, num_workers=4)

    model.eval()
    results = defaultdict(list)
    for batch in tqdm(validation_loader, total=len(validation_loader)):
        with torch.no_grad():
            inputs = {}
            for key in model_args.model_inputs:
                inputs[key] = batch[key][0].to(device)
            generations = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=model_args.max_new_tokens,
                num_beams=model_args.num_beams,
                temperature=model_args.temperature,
                no_repeat_ngram_size=model_args.no_repeat_ngram_size,
                top_p=model_args.top_p,
            )
            generations.to(torch.device("cpu"))

            for metric in metrics:
                res = metric.compute(generations)
                for key, val in res.items():
                    results[key].append(val)

    for key, val in results.items():
        print(key, ": ", np.mean(val))


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).absolute()))
    else:
        args = parser.parse_args_into_dataclasses()

    raise SystemExit(main(*args))
