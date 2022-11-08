"""
    train_pronunciation_embeddings.py
    Created by Pierre Tessier
    10/19/22 - 4:50 PM
    Description:
    Train word2vec like embeddings for pronunciation and stress patterns on the Gutenberg Poetry Corpus.
 """
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from src import CMUDictionary, CMULinker, build_dataset, load_gutenberg, PronunciationTokenizer, train


def main(args):
    model_name = args.model
    pronunciation_dataset_path = Path(args.dataset)
    pronunciation_embeddings_path = Path("etc/pronunciation_embeddings") / "p_{}_{}_s_{}_{}_sg_{}_negative_{}".format(
        args.pronunciation[0], args.pronunciation[1], args.stress[0], args.stress[1], args.sg, args.negative)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    corpus = load_gutenberg(n_lines=None)
    dataset = build_dataset(corpus, tokenizer, batched=False, num_proc=1, padding=False, truncation=False)

    tokenizer_p.dataset_to_files(dataset, pronunciation_dataset_path)
    train(tokenizer_p, pronunciation_dataset_path, pronunciation_embeddings_path,
          model_p={"vector_size": args.pronunciation[0], "window": args.pronunciation[1]},
          model_s={"vector_size": args.stress[0], "window": args.stress[1]},
          sg=args.sg, hs=0, negative=args.negative, workers=args.workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train word2vec like embeddings for pronunciation and stress patterns on the Gutenberg Poetry Corpus.")
    parser.add_argument("-d", "--dataset", type=str, help="Path to the dataset to use.")
    parser.add_argument("-m", "--model", default="EleutherAI/gpt-neo-125M",
                        help="Transformer model used (to find a matching tokenizer).")
    parser.add_argument("-p", "--pronunciation", default=[32, 8], nargs=2, type=int,
                        help="Vector-size and window-size to use for the pronunciation embeddings.")
    parser.add_argument("-s", "--stress", default=[32, 8], nargs=2, type=int,
                        help="Vector-size and window-size to use for the stress embeddings.")
    parser.add_argument("--sg", type=int, default=1, help="1 for skip-gram, 0 for CBOW.")
    parser.add_argument("--negative", type=int, default=5, help="Number of negative samples to use.")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for the word2vec algorithm.")

    args = parser.parse_args()
    raise SystemExit(main(args))
