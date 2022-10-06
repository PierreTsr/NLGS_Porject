"""
    cum_tools.py
    Created by pierre
    10/5/22 - 6:32 PM
    Description:
    # Enter file description
 """
import re
from typing import Optional

phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
            "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z",
            "ZH"]
phonemes_dict = {pho: idx for idx, pho in enumerate(phonemes)}
stress = [0, 1, 2]

phoneme_pattern = re.compile(r"^([A-Z]+)([0-2]*)$")
multi_word_pattern = re.compile(r"^(.+)\((\d)\)$")


def split_phoneme(phoneme: str) -> tuple[str, Optional[int]]:
    a, b = phoneme_pattern.match(phoneme).groups()
    return a, int(b) if b else None


def load_cmu(path: str = "../data/cmu_dict/cmudict-0.7b") -> dict[str, list[tuple[list[int], list[Optional[int]]]]]:
    d = {}
    with open(path, "r", encoding="us-ascii") as file:
        line = file.readline()
        while line[:3] == ";;;":
            line = file.readline()
        while line != "\n":
            try:
                line = file.readline()
                line = line.split()
                word = line[0].lower()
            except UnicodeError:
                continue
            except IndexError:
                break
            p = []
            s = []
            for pho in line[1:]:
                a, b = split_phoneme(pho)
                p.append(phonemes_dict[a])
                s.append(b)
            if multi_word_pattern.match(word):
                word = multi_word_pattern.match(word).groups()[0]
                d[word].append((p, s))
            else:
                d[word] = [(p, s)]
    return d


def rhyming(word_a: str, word_b: str) -> bool:
    pass


def get_pronunciation(tokens: list[str]) -> list[int]:
    pass


def get_stress(tokens: list[str]) -> list[int]:
    pass
