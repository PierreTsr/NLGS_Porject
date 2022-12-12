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
import wandb
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
    rhyme_mixin_coeff: Optional[float] = field(
        default=1.0
    )
    alliteration_mixin_coeff: Optional[float] = field(
        default=1.0
    )
    meter_mixin_coeff: Optional[float] = field(
        default=1.0
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


def main(model_args: ModelArguments, data_args: DataTrainingArguments, sweep=False):
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
        raise (NameError("Invalid value for meter."))

    meter = MeterMetrics(linker, tokenizer_p, patterns=patterns, verbose=False)
    rhyme = RhymingMetrics(linker, tokenizer_p, verbose=False)
    alliteration = AlliterationMetrics(linker, tokenizer_p, verbose=False)
    distinct = DistinctMetrics(tokenizer, 4, verbose=False)

    metrics = [meter, rhyme, alliteration, distinct]

    stop = [VerseCountStoppingCriteria(4)]
    proc = LogitsProcessorList([
        TopKLogitsWarper(50),
        NoRepeatNGramLogitsProcessor(4),
        MeterLogitsProcessor(meter.damerauLevenshtein_all_vocab_score, model_args.meter_mixin_coeff),
        AStarLogitsProcessor(
            model,
            [
                RhymeAuxLogitsProcessor(rhyme.score_generation, model_args.rhyme_mixin_coeff),
                AlliterationAuxLogitsProcessor(alliteration.score_generation, model_args.alliteration_mixin_coeff)
            ],
            sample=True,
            top_k=10,
            num_beams=5,
            max_new_tokens=10,
            n_newlines=1,
            temperature=0.5,
        )
    ])

    sampler = RandomSampler(dataset)
    validation_loader = DataLoader(dataset, sampler=sampler, num_workers=0)

    filename = None
    if model_args.save_to_file:
        model_name = Path(model_args.model_name_or_path).name
        data_name = Path(data_args.dataset_path).name
        generation_dir = Path(model_args.save_to_file)
        generation_dir.mkdir(exist_ok=True)
        filename = generation_dir / "generations_{}_{}_mixins_{:.0e}_{:.0e}_{:.0e}_t_{:.1f}_nb_{}.txt".format(
            model_name,
            data_name,
            model_args.alliteration_mixin_coeff,
            model_args.meter_mixin_coeff,
            model_args.rhyme_mixin_coeff,
            model_args.temperature,
            model_args.num_beams,
        )
        with open(filename, "w") as file:
            file.write("{} samples generated by {} on {}\n\n\n".format(data_args.n_samples,
                                                                           model_args.model_name_or_path, datetime.now()))
            file.write("With configuration: {}".format(str(model_args)))

    model.eval()
    results = defaultdict(list)
    ct = 0
    n = data_args.n_samples
    quatrain = []
    for sample in tqdm(validation_loader, total=n, desc="Quatrain generation"):
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
                logits_processor=proc,
                stopping_criteria=stop,
                renormalize_logits=True,
                pad_token_id=tokenizer.pad_token_id
            )
            generations.to(torch.device("cpu"))

        mask = generations == newline_token_id
        new_lines = (mask[:, 1:] ^ mask.roll(1, dims=1)[:, 1:]) & mask[:, 1:]
        count = torch.max(torch.sum(new_lines, dim=1)).item()
        is_quatrain = 3 <= count <= 4
        quatrain.append(is_quatrain)

        logs = {
            "quatrain": is_quatrain
        }
        for metric in metrics:
            res = metric.compute(generations)
            logs.update(res)
            for key, val in res.items():
                results[key].append(val)

        if sweep:
            wandb.log(logs)

        if filename is not None:
            with open(filename, "a") as file:
                file.write(tokenizer.batch_decode(generations, skip_special_tokens=True)[0] + "\n\n<SEP>\n\n")

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

def get_run_fn(model_args, data_args):
    def run(config=None):
        with wandb.init(config=config):
            config = wandb.config
            model_args.model_name_or_path = config.model
            data_args.dataset_path = config.dataset
            model_args.num_beams = config.num_beams
            model_args.temperature = config.temperature
            model_args.meter_mixin_coeff = config.meter_mixin_coeff
            model_args.rhyme_mixin_coeff = config.rhyme_mixin_coeff
            model_args.alliteration_mixin_coeff = config.alliteration_mixin_coeff
            if Path(data_args.dataset_path).name == "pentameter_prompts":
                model_args.meter = "iambic_pentameter"
            elif Path(data_args.dataset_path).name == "mixed_meter_prompts":
                model_args.meter = "mixed_meter"
            main(model_args, data_args, sweep=True)

    return run


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).absolute()))
        raise SystemExit(main(model_args, data_args))
    elif len(sys.argv) == 1:
        model_args, data_args = parser.parse_json_file("etc/config/evaluation_config_vanilla.json")
        run_fn = get_run_fn(model_args, data_args)
        sweep_config = {
            "method": "random",
            "metric": {
                "name": "perfect_rhymes",
                "goal": "maximize"
            },
            "parameters": {
                "model": {
                    "values": [
                        "etc/gpt-neo-2.7B-fine-tuned",
                        "EleutherAI/gpt-neo-2.7B",
                        "etc/gpt-neo-2.7B-custom",
                    ]
                },
                "dataset": {
                    "values": [
                        "data/datasets/pentameter_prompts",
                    ]
                },
                "meter_mixin_coeff": {
                    "values": [1e-3, 1e-2, 1e-1, 1, 10]
                },
                "rhyme_mixin_coeff": {
                    "values": [1e-3, 1e-2, 1e-1, 1, 10]
                },
                "alliteration_mixin_coeff": {
                    "values": [1e-3, 1e-2, 1e-1, 1, 10]
                },
                "num_beams": {
                    "values": [1, 5, 10, 20]
                },
                "temperature": {
                    "values": [.3, .5, .7, .9, 1.1, 1.3]
                }
            }
        }
        wandb.login()
        sweep_id = wandb.sweep(sweep_config, project="NLGS_Project_decoding")
        wandb.agent(sweep_id, run_fn, count=200)
        raise (SystemExit(0))
    else:
        parser.parse_args()
