"""
    gutenberg_explorer.py
    Created by Pierre Tessier
    10/18/22 - 11:38 PM
    Description:
    # Enter file description
 """
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from gutenberg_poetry_corpus import build_corpus
from src.cmu_tools import CMULinker, CMUDictionary
from src.pronunciation_embeddings import PronunciationTokenizer

def main(argv=None):
    corpus = build_corpus()
    cmu = CMUDictionary()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    n_lines = []
    n_tokens = []
    n_tokens_avg = []
    n_phonemes_word = []
    n_phonemes_line = []
    for lines in tqdm(corpus.values()):
        n_lines.append(len(lines))
        s_tok = 0
        for l in lines:
            tok = tokenizer(l)
            n_tok = len(tok["input_ids"])
            s_tok += n_tok
            n_tokens.append(n_tok)
            s_pho = 0
            for word in tok["input_ids"]:
                try:
                    p, s = linker.get_pronunciation(word)
                    n_phonemes_word.append(len(p))
                    s_pho += len(p)
                except KeyError:
                    continue
            n_phonemes_line.append(s_pho)
        s_tok = s_tok / n_lines[-1]
        n_tokens_avg.append(s_tok)

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()
    log_bins = np.logspace(0, np.log10(max(n_lines)), 50)
    axs[0].hist(n_lines, log_bins)
    axs[0].axvline(x=np.mean(n_lines), color="r", label="mean")
    axs[0].legend()
    axs[0].set_xscale("log")
    axs[0].set_title("Number of lines per document")

    axs[1].hist(n_tokens, bins=range(17))
    axs[1].axvline(x=np.mean(n_tokens), color="r", label="mean")
    axs[1].legend()
    axs[1].set_title("Number of tokens per line")

    axs[2].hist(n_tokens_avg, bins=20)
    axs[2].axvline(x=np.mean(n_tokens_avg), color="r", label="mean")
    axs[2].legend()
    axs[2].set_title("Average number of tokens per line in a document")

    axs[3].hist(n_phonemes_word, bins=20)
    axs[3].axvline(x=np.mean(n_phonemes_word), color="r", label="mean")
    axs[3].legend()
    axs[3].set_title("Average number of phonemes per word in a document")

    axs[4].hist(n_phonemes_line, bins=20)
    axs[4].axvline(x=np.mean(n_phonemes_line), color="r", label="mean")
    axs[4].legend()
    axs[4].set_title("Average number of phonemes per line in a document")

    plt.tight_layout()
    plt.show()

    exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
