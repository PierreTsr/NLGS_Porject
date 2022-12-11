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
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM

from src import load_dataset, CMUDictionary, CMULinker, PronunciationTokenizer, VerseCountStoppingCriteria
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
    batch_size: int = field(
        default=1,
        metadata={"help": "Number of generations to process at once."}
    )
    n_samples: int = field(
        default=1000,
        metadata={"help": "Number of prompts to sample for generations."}
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
    save_to_file: Optional[str] = field(
        default=None,
        metadata={"help": "folder to save the generations to."}
    )
    meter: Optional[str] = field(
        default="iambic_pentameter",
        metadata={"help": "Type of meter to use (iambic_pentameter or mixed_meter)"}
    )


pentameter_patterns = [
    # pentameter
    "x/x/x/x/x/",
    "/xx/x/x/x/",
    "xx/x/x/x/x/",
]

mixed_meter_patterns = pentameter_patterns + [
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


def main(model_args: ModelArguments, data_args: DataTrainingArguments):
    dataset = load_dataset(data_args.dataset_path)["validation"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    newline_token_id = 198

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    device = torch.device("cpu")
    if not model_args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    if model_args.meter == "iambic_pentameter":
        patterns = pentameter_patterns
    elif model_args.meter == "mixed_meter":
        patterns = mixed_meter_patterns
    else:
        raise(NameError("Invalid value for meter."))

    metrics = [
        RhymingMetrics(linker, tokenizer_p, verbose=False),
        AlliterationMetrics(linker, tokenizer_p, verbose=False),
        MeterMetrics(linker, tokenizer_p, patterns=patterns, verbose=False),
        DistinctMetrics(tokenizer, 4, verbose=False)
    ]
    stop = [VerseCountStoppingCriteria(4)]

    sampler = SequentialSampler(dataset)
    validation_loader = DataLoader(dataset, sampler=sampler, num_workers=0)

    filename = None
    if model_args.save_to_file:
        model_name = Path(model_args.model_name_or_path).name
        data_name = Path(data_args.dataset_path).name
        generation_dir = Path(model_args.save_to_file)
        generation_dir.mkdir(exist_ok=True)
        filename = generation_dir / "generations_{}_{}.txt".format(model_name, data_name)
        with open(filename, "w") as file:
            file.write("{} samples generated by {} on {}\n\n\n".format(data_args.n_samples,
                                                                           model_args.model_name_or_path, datetime.now()))

    model.eval()
    results = defaultdict(list)
    ct = 0
    n = data_args.n_samples
    quatrain = []
    for sample in tqdm(validation_loader, total=n):
        if ct == n:
            break
        ct += 1
        with torch.no_grad():
            inputs, attention = sample["input_ids"], sample["attention_mask"]
            inputs = inputs[attention].view(1, -1).to(device)
            generations = model.generate(
                inputs,
                max_new_tokens=model_args.max_new_tokens,
                num_beams=model_args.num_beams,
                temperature=model_args.temperature,
                no_repeat_ngram_size=model_args.no_repeat_ngram_size,
                stopping_criteria=stop,
                top_p=model_args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            generations.to(torch.device("cpu"))

        for metric in metrics:
            res = metric.compute(generations)
            for key, val in res.items():
                results[key].append(val)

        if filename is not None:
            with open(filename, "a") as file:
                file.write(tokenizer.batch_decode(generations, skip_special_tokens=True)[0] + "\n\n<SEP>\n\n")

        mask = generations == newline_token_id
        new_lines = (mask[:, 1:] ^ mask.roll(1, dims=1)[:, 1:]) & mask[:, 1:]
        count = torch.max(torch.sum(new_lines, dim=1)).item()
        quatrain.append(count >= 3)

    quatrain = np.array(quatrain, dtype=bool)
    print("Generation with " + model_args.model_name_or_path)
    print("Correct quatrains: {}".format(np.sum(quatrain)))
    print("Metrics results total:")
    for key, val in results.items():
        print("   {}: {}".format(key, np.mean(np.array(val))))
    print("\nMetrics results quatrains:")
    for key, val in results.items():
        print("   {}: {}".format(key, np.mean(np.array(val)[quatrain])))

    if filename is not None:
        with open(filename, "a") as file:
            file.write("Correct quatrains: {}\n\n".format(np.sum(quatrain)))
            file.write("Metrics results total:\n")
            for key, val in results.items():
                file.write("   {}: {}\n".format(key, np.mean(np.array(val))))
            file.write("\nMetrics results quatrains:\n")
            for key, val in results.items():
                file.write("   {}: {}\n".format(key, np.mean(np.array(val)[quatrain])))


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).absolute()))
    else:
        args = parser.parse_args_into_dataclasses()

    raise SystemExit(main(*args))
