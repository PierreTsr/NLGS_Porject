"""
    select_clean_prompts.py
    Created by Pierre Tessier
    12/1/22 - 11:58 AM
    Description:
    # Enter file description
 """
import argparse
import sys

from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src import MeterMetrics, CMUDictionary, CMULinker, PronunciationTokenizer

correct_patterns = [
    "x/x/x/x/x/"
]

dot_id = 13
newline_id = 198
unk_id = 50256

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    dataset = load_from_disk(args.dataset)["validation"]
    metric = MeterMetrics(linker, tokenizer_p, patterns=correct_patterns, verbose=False)

    prompt_doc = []
    prompt_pos = []
    for doc_idx, doc in tqdm(enumerate(dataset), total=len(dataset)):
        start = 0
        sentence_beginning = True
        seen_dot = False
        for idx, token in enumerate(doc["input_ids"]):
            if token == unk_id:
                continue
            elif token == dot_id:
                seen_dot = True
            elif token == newline_id:
                if sentence_beginning:
                    candidate = doc["input_ids"][start:idx]
                    _, meter_val, _ = metric.max_damerauLevenshtein(candidate)
                    if meter_val >= args.threshold:
                        prompt_doc.append(doc_idx)
                        prompt_pos.append((start, idx))
                start = idx + 1
                sentence_beginning = seen_dot
            else:
                seen_dot = False

    prompts = []
    for doc_idx, (start, end) in tqdm(zip(prompt_doc, prompt_pos), total=len(prompt_doc)):
        doc = dataset[doc_idx]
        line = doc["input_ids"][start:end]
        entry = { key: val[start:end] for key, val in doc.items()}
        # entry["text"] = tokenizer.decode(line)
        prompts.append(entry)
    prompts = Dataset.from_list(prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Path to the dataset to use.")
    parser.add_argument("-t", "--threshold", type=float, default="Threshold value for the meter metric.")
    args = parser.parse_args()

    raise SystemExit(main(args))
