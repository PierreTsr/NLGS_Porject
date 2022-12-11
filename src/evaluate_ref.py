"""
    evaluate_model.py
    Created by Pierre Tessier
    10/30/22 - 1:00 PM
    Description:
    # Enter file description
 """
import sys
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_from_disk
from transformers import HfArgumentParser, AutoTokenizer

from src import load_dataset, CMUDictionary, CMULinker, PronunciationTokenizer
from src.metrics import RhymingMetrics, AlliterationMetrics, MeterMetrics, DistinctMetrics


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

correct_patterns = [
    # pentameter
    "x/x/x/x/x/",
    "/xx/x/x/x/",
    "xx/x/x/x/x/",

    # tetrameter
    "/x/x/x/x",
    "xx/xx/xx/xx/",
    "x/x/x/x/",
    "/x/x/x/",

    # trimeter
    "x/x/x/",

    # hexameter
    "x/x/x/x/x/x/",
    "/xx/xx/xx/xx/xx/xx",
]


def main(data_args: DataTrainingArguments):
    dataset = load_from_disk(data_args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    metrics = [
        RhymingMetrics(linker, tokenizer_p, verbose=True),
        AlliterationMetrics(linker, tokenizer_p, verbose=True),
        MeterMetrics(linker, tokenizer_p,patterns=correct_patterns, verbose=True),
        DistinctMetrics(tokenizer, 4, verbose=True)
    ]

    results = {}
    for metric in metrics:
        results |= metric.compute(dataset["validation"]["input_ids"])

    print(results)


if __name__ == "__main__":
    parser = HfArgumentParser(DataTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).absolute()))
    else:
        args = parser.parse_args_into_dataclasses()

    raise SystemExit(main(*args))
