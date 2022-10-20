"""
    preprocess_dataset.py
    Created by Pierre Tessier
    10/20/22 - 1:10 PM
    Description:
    Script to pre-process and build the Gutenberg Poetry Corpus into a HuggingFace's dataset.
 """
from argparse import ArgumentParser

from transformers import AutoTokenizer

from src.cmu_tools import CMUDictionary, CMULinker
from src.gutenberg_tools import load_gutenberg
from src.poetry_datasets import build_dataset
from src.pronunciation_embeddings import PronunciationTokenizer


def main(args):
    corpus = load_gutenberg(n_lines=args.n_lines, stride=args.stride)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.pronunciation:
        cmu = CMUDictionary()
        linker = CMULinker(tokenizer, cmu)
        tokenizer_p = PronunciationTokenizer(linker, tokenizer)

        data_path = "data/datasets/gutenberg_pronunciation_chunked_{}_{}_{}".format(
            args.n_lines, args.stride, args.max_length)

        dataset = build_dataset(
            corpus,
            tokenizer,
            tokenizer_p=tokenizer_p,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_proc=args.workers,
            padding='max_length',
            truncation=True
        )
    else:
        data_path = "data/datasets/gutenberg_chunked_{}_{}_{}".format(args.n_lines, args.stride, args.max_length)

        dataset = build_dataset(
            corpus,
            tokenizer,
            batch_size=args.batch_size,
            num_proc=args.workers,
            padding='max_length',
            truncation=True
        )
    dataset.save_to_disk(data_path)
    exit(0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Script to pre-process and build the Gutenberg Poetry Corpus into a "
                                        "HuggingFace's dataset.")
    parser.add_argument("model_name", type=str, help="Model name whose tokenizer should be used.")
    parser.add_argument("-p", "--pronunciation", action="store_true", help="Include the pronunciation representation.")
    parser.add_argument("-n", "--n_lines", default=128, type=int, help="Number of lines per chunk of corpus.")
    parser.add_argument("-s", "--stride", default=92, type=int, help="Stride when splitting the corpus in chunks.")
    parser.add_argument("-m", "--max_length", default=8, type=int, help="Truncation length of the pronunciation "
                                                                        "representations.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for the dataset processing.")
    parser.add_argument("--workers", default=1, type=int, help="Number of workers for the dataset processing.")
    args = parser.parse_args()
    raise SystemExit(main(args))
