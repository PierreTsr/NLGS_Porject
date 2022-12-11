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

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TopKLogitsWarper, \
    NoRepeatNGramLogitsProcessor

from src import load_dataset, CMUDictionary, CMULinker, PronunciationTokenizer, VerseCountStoppingCriteria, \
    MeterLogitsProcessor, AStarLogitsProcessor, RhymeAuxLogitsProcessor, AlliterationAuxLogitsProcessor
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
        metadata={"help": "File to save the generations to."}
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


def main(model_args: ModelArguments, data_args: DataTrainingArguments):
    global num_beams
    global temperature
    global meter_mixin_coeff
    global rhyme_mixin_coeff
    global alliteration_mixin_coeff

    dataset = load_dataset(data_args.dataset_path)["validation"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    device = torch.device("cpu")
    if not model_args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    meter = MeterMetrics(linker, tokenizer_p, patterns=correct_patterns, verbose=False)
    rhyme = RhymingMetrics(linker, tokenizer_p, verbose=False)
    alliteration = AlliterationMetrics(linker, tokenizer_p, verbose=False)
    distinct = DistinctMetrics(tokenizer, 4, verbose=False)

    metrics = [meter, rhyme, alliteration, distinct]

    stop = [VerseCountStoppingCriteria(4)]
    proc = LogitsProcessorList([
        TopKLogitsWarper(50),
        NoRepeatNGramLogitsProcessor(4),
        MeterLogitsProcessor(meter.damerauLevenshtein_all_vocab_score, meter_mixin_coeff),
        AStarLogitsProcessor(model, [
            RhymeAuxLogitsProcessor(rhyme.score_generation, rhyme_mixin_coeff),
            AlliterationAuxLogitsProcessor(alliteration.score_generation, alliteration_mixin_coeff)
        ], top_k=10, max_new_tokens=10, n_newlines=1, num_beams=5, temperature=0.3, sample=True)
    ])

    sampler = RandomSampler(dataset)
    validation_loader = DataLoader(dataset, sampler=sampler, num_workers=0)

    save = False
    if model_args.save_to_file:
        save = True
        with open(model_args.save_to_file, "w") as file:
            file.write("{} samples generated by {} on {}\n\n\n".format(data_args.n_samples,
                                                                       model_args.model_name_or_path, datetime.now()))

    model.eval()
    results = defaultdict(list)
    ct = 0
    n = data_args.n_samples
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
                num_beams=num_beams,
                temperature=temperature,
                logits_processor=proc,
                stopping_criteria=stop,
                renormalize_logits=True,
                pad_token_id=tokenizer.pad_token_id
            )
            generations.to(torch.device("cpu"))

        for metric in metrics:
            res = metric.compute(generations)
            for key, val in res.items():
                results[key].append(val)

        if save:
            with open(model_args.save_to_file, "a") as file:
                file.write(tokenizer.batch_decode(generations, skip_special_tokens=True)[0] + "\n\n")

    for txt in tokenizer.batch_decode(generations, skip_special_tokens=True):
        print(txt, "\n")

    for key, val in results.items():
        print(key, ": ", np.mean(val))
    return results


def get_run_fn(args):
    def run(config=None):
        global num_beams
        global temperature
        global meter_mixin_coeff
        global rhyme_mixin_coeff
        global alliteration_mixin_coeff

        with wandb.init(config=config):
            config = wandb.config

            num_beams = config.nums_beams
            temperature = config.temperature
            meter_mixin_coeff = config.meter_mixin_coeff
            rhyme_mixin_coeff = config.rhyme_wixin_coeff
            alliteration_mixin_coeff = config.alliteration_mixin_coeff
            results = main(args)
            wandb.log(results)
    return run



if __name__ == "__main__":
    num_beams = 5
    temperature = 0.7
    meter_mixin_coeff = 0.1
    rhyme_mixin_coeff = 0.1
    alliteration_mixin_coeff = 0.1

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).absolute()))
        raise SystemExit(main(*args))
    else:
        args = parser.parse_args_into_dataclasses("etc/config/evaluation_config_vanilla.json")
        run_fn = get_run_fn(args)
        sweep_config = {
            "method": "random",
            "metric": {
                "name": "perfect_rhymes",
                "goal": "maximize"
            },
            "parameters": {
                "num_beams": {
                    "values": [1, 5, 10, 20]
                },
                "temperature": {
                    "values": [.3, .5, .7, .9, 1.1, 1.3]
                },
                "meter_mixin_coeff": {
                    "values": [1e-3, 1e-2, 1e-1, 1, 10]
                },
                "rhyme_mixin_coeff": {
                    "values": [1e-3, 1e-2, 1e-1, 1, 10]
                },
                "alliteration_mixin_coeff": {
                    "values": [1e-3, 1e-2, 1e-1, 1, 10]
                }
            }
        }
        sweep_id = wandb.sweep(sweep_config, project="NLGS_Project")
        wandb.agent(sweep_id, run_fn, count=100)
