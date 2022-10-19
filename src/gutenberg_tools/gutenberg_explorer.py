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

from src.gutenberg_tools.gutenberg_poetry_corpus import build_corpus


def main(argv=None):
    corpus = build_corpus()
    n_lines = []
    n_tokens = []
    n_tokens_avg = []
    for lines in tqdm(corpus.values()):
        n_lines.append(len(lines))
        s = 0
        for l in lines:
            n = len(l.split())
            n_tokens.append(n)
            s += n
        s = s / n_lines[-1]
        n_tokens_avg.append(s)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
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

    plt.tight_layout()
    plt.show()

    exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
