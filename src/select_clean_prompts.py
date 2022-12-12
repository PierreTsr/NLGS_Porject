"""
    select_clean_prompts.py
    Created by Pierre Tessier
    12/1/22 - 11:58 AM
    Description:
    # Enter file description
 """
import argparse

from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

from src import MeterMetrics, CMUDictionary, CMULinker, PronunciationTokenizer

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

dot_id = 13
newline_id = 198
unk_id = 50256

def filter_prompts(dataset, metric, tokenizer):
    prompt_pos = []
    for doc_idx, doc in tqdm(enumerate(dataset), total=len(dataset)):
        start = 0
        sentence_beginning = True
        seen_dot = False
        prompt_pos.append([])
        for idx, token in enumerate(doc["input_ids"]):
            if token == unk_id:
                continue
            elif token == dot_id:
                seen_dot = True
            elif token == newline_id:
                if sentence_beginning:
                    candidate = doc["input_ids"][start:idx]
                    _, meter_val, _ = metric.min_damerauLevenshtein(candidate)
                    if meter_val <= args.threshold:
                        prompt_pos[-1].append((start, idx))
                start = idx + 1
                sentence_beginning = seen_dot
            else:
                seen_dot = False

    prompts = []
    for doc_idx, positions in tqdm(enumerate(prompt_pos), total=len(prompt_pos)):
        doc = dataset[doc_idx]
        for (start, end) in positions:
            line = doc["input_ids"][start:end]
            entry = {key: val[start:end] for key, val in doc.items()}
            entry["text"] = tokenizer.decode(line)
            prompts.append(entry)
    prompts = Dataset.from_list(prompts).shuffle(seed=42)
    return prompts

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    dataset = load_from_disk(args.dataset)
    if args.meter == "iambic_pentameter":
        patterns = pentameter_patterns
    elif args.meter == "mixed_meter":
        patterns = mixed_meter_patterns
    else:
        raise(ValueError("Incorrect --meter value."))
    metric = MeterMetrics(linker, tokenizer_p, patterns=patterns, verbose=False)

    prompts = DatasetDict({
        "test": filter_prompts(dataset["test"], metric, tokenizer),
        "validation": filter_prompts(dataset["validation"], metric, tokenizer)
    })

    prompts.save_to_disk(args.dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Path to the dataset to use.")
    parser.add_argument("-t", "--threshold", type=float, default="Threshold value for the meter metric.")
    parser.add_argument("-m", "--meter", type=str, choices=("mixed_meter", "iambic_pentameter"), help="Filter to apply on the meter of the selected prompts.")
    parser.add_argument("--dest", type=str, help="Path of the new dataset to save.")
    args = parser.parse_args()

    raise SystemExit(main(args))
