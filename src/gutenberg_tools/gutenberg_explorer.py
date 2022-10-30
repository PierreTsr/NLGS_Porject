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
    n_stress_word = []
    n_stress_line = []
    for lines in tqdm(corpus.values()):
        n_lines.append(len(lines))
        s_tok = 0
        for l in lines:
            tok = tokenizer(l)
            n_tok = len(tok["input_ids"])
            s_tok += n_tok
            n_tokens.append(n_tok)
            s_pho = 0
            s_stress = 0
            for word in tok["input_ids"]:
                try:
                    p, s = linker.get_pronunciation(word)
                    s = sum([x >= 1 for x in s])
                    n_phonemes_word.append(len(p))
                    n_stress_word.append(s)
                    s_pho += len(p)
                    s_stress += s
                except KeyError:
                    continue
            n_phonemes_line.append(s_pho)
            n_stress_line.append(s_stress)
        s_tok = s_tok / n_lines[-1]
        n_tokens_avg.append(s_tok)

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()
    log_bins = np.logspace(0, np.log10(max(n_lines)), 50)

    axs[0].hist(n_lines, log_bins, zorder=3, edgecolor="black")
    axs[0].axvline(x=np.mean(n_lines), color="r", label="mean", zorder=3)
    axs[0].grid(True, axis="y", zorder=0)
    axs[0].legend()
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Number of lines")
    axs[0].set_ylabel("Number of documents")
    axs[0].set_title("Number of lines per document")

    axs[1].hist(n_tokens, bins=range(25), zorder=3, edgecolor="black")
    axs[1].axvline(x=np.mean(n_tokens) + 0.5, color="r", label="mean", zorder=3)
    ticks = range(0, 25, 2)
    axs[1].set_xticks(ticks=np.array(ticks) + 0.5, labels=ticks)
    axs[1].grid(True, axis="y", zorder=0)
    axs[1].legend()
    axs[1].set_xlabel("Number of tokens")
    axs[1].set_ylabel("Number of lines")
    axs[1].set_title("Number of tokens per line")

    axs[2].hist(n_phonemes_word, bins=range(12), zorder=3, edgecolor="black", density=True)
    axs[2].axvline(x=np.mean(n_phonemes_word) + 0.5, color="r", label="mean", zorder=3)
    ticks = range(0, 12)
    axs[2].set_xticks(ticks=np.array(ticks) + 0.5, labels=ticks)
    axs[2].grid(True, axis="y", zorder=0)
    axs[2].legend()
    axs[2].set_xlabel("Frequencies")
    axs[2].set_ylabel("Number of phonemes per word")
    axs[2].set_title("Number of phonemes per word")

    axs[4].hist(n_phonemes_line, bins=range(50), zorder=3, edgecolor="black", density=True)
    axs[4].axvline(x=np.mean(n_phonemes_line) + 0.5, color="r", label="mean", zorder=3)
    ticks = range(0, 50, 4)
    axs[4].set_xticks(ticks=np.array(ticks) + 0.5, labels=ticks)
    axs[4].grid(True, axis="y", zorder=0)
    axs[4].legend()
    axs[4].set_xlabel("Frequencies")
    axs[4].set_ylabel("Number of phonemes per line")
    axs[4].set_title("Number of phonemes per line")

    axs[3].hist(n_stress_word, bins=range(6), zorder=3, edgecolor="black", density=True)
    axs[3].axvline(x=np.mean(n_stress_word) + 0.5, color="r", label="mean", zorder=3)
    ticks = range(6)
    axs[3].set_xticks(ticks=np.array(ticks) + 0.5, labels=ticks)
    axs[3].grid(True, axis="y", zorder=0)
    axs[3].legend()
    axs[3].set_xlabel("Frequencies")
    axs[3].set_ylabel("Number of syllables per word")
    axs[3].set_title("Number of syllables per word")

    axs[5].hist(n_stress_line, bins=range(20), zorder=3, edgecolor="black", density=True)
    axs[5].axvline(x=np.mean(n_stress_line) + 0.5, color="r", label="mean", zorder=3)
    ticks = range(0, 22, 2)
    axs[5].set_xticks(ticks=np.array(ticks) + 0.5, labels=ticks)
    axs[5].grid(True, axis="y", zorder=0)
    axs[5].legend()
    axs[5].set_xlabel("Frequencies")
    axs[5].set_ylabel("Number of syllables per line")
    axs[5].set_title("Number of syllables per line")

    plt.tight_layout()
    plt.show()

    exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
