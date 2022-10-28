"""
    cum_tools.py
    Created by pierre
    10/5/22 - 6:32 PM
    Description:
    Set of tools to load and manipulate the CMU dictionary of pronunciation.
 """
import re
import warnings

CMU_type = dict[str, list[tuple[list[int], list[int]]]]

phoneme_pattern = re.compile(r"^([A-Z]+)([0-2]*)$")
multi_word_pattern = re.compile(r"^(.+)\((\d)\)$")


class CMUDictionary:
    phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
                "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y",
                "Z", "ZH"]
    phonemes_dict = {pho: idx for idx, pho in enumerate(phonemes)}
    stress = [-1, 0, 1, 2]
    vowels = {idx for idx, pho in enumerate(phonemes) if pho[0] in {"A", "E", "I", "O", "U", "Y"}}
    consonants = {idx for idx, pho in enumerate(phonemes) if pho[0] not in {"A", "E", "I", "O", "U", "Y"}}

    def __init__(self, path: str = "data/cmu_dict/cmudict-0.7b") -> None:
        """
        Create a tool to manipulate the CMU dictionary of pronunciation.

        :param path: Path to the CMU dictionary data file.
        :type path: str
        """
        self.cmu_dictionary = {}
        self.rhyme_dictionary = {}
        self.load_cmu_dict(path)
        self.build_rhyme_dict()

    def load_cmu_dict(self, path: str) -> int:
        """
        Load the CMU dictionary as a Python dictionary.

        :param path: Path to the CMU dictionary data file.
        :type path: str
        :return: 0 if the creation was successful
        :rtype: int
        """
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
                    warnings.warn("Detected a corrupted entry, please check file encoding.", UnicodeWarning)
                    continue
                except IndexError:
                    break
                p = []
                s = []
                for pho in line[1:]:
                    a, b = split_phoneme(pho)
                    p.append(self.phonemes_dict[a])
                    s.append(b + 1)
                if multi_word_pattern.match(word):
                    word = multi_word_pattern.match(word).groups()[0]
                    self.cmu_dictionary[word].append((p, s))
                else:
                    self.cmu_dictionary[word] = [(p, s)]
        return 0

    def build_rhyme_dict(self) -> int:
        """
        Create a reverse dictionary with the rhyming part of each word as keys.

        :return: 0 if the creation was successful.
        :rtype: int
        """
        assert self.cmu_dictionary
        for word, pronunciations in self.cmu_dictionary.items():
            for (phonemes, stress) in pronunciations:
                key = get_rhyming_part(phonemes, stress)
                if key not in self.rhyme_dictionary:
                    self.rhyme_dictionary[key] = []
                self.rhyme_dictionary[key].append(word)
        return 0

    def is_rhyming(self, word_a: str, word_b: str) -> bool:
        """
        Check if two words are rhyming perfectly.

        :param word_a: First word.
        :type word_a: str
        :param word_b: Second word.
        :type word_b: str
        :return: True if the two words form a perfect rhyme.
        :rtype: bool
        """
        pronunciations_a = self.cmu_dictionary[word_a]
        rhymes_a = [get_rhyming_part(*pronunciation) for pronunciation in pronunciations_a]
        rhymes_a = {(s, *p) for s, p in rhymes_a}
        pronunciations_b = self.cmu_dictionary[word_b]
        rhymes_b = [get_rhyming_part(*pronunciation) for pronunciation in pronunciations_b]
        rhymes_b = {(s, *p) for s, p in rhymes_b}
        res = rhymes_a.intersection(rhymes_b)
        return bool(res)


def split_phoneme(phoneme: str) -> tuple[str, int]:
    """
    Separate the phoneme from the stress information in ARPAbet notation.

    :param phoneme: Phoneme as found in the CMU dictionary.
    :type phoneme: str
    :return: Tuple with the phoneme in first position, and the stress information in second.
    :rtype: tuple[str, int]
    """
    a, b = phoneme_pattern.match(phoneme).groups()
    return a, int(b) if b else -1


def get_rhyming_part(phonemes: list[int], stress: list[int]) -> tuple[int, ...]:
    """
    Extract the rhyming part from pronunciation pattern.

    :param phonemes: list of phonemes without stress information.
    :type phonemes: list[str]
    :param stress: list of stress information for each phoneme.
    :type stress: list[int]
    :return: tuple with the last stress value, and the phonemes forming the rhyme.
    :rtype: tuple[int]
    """
    try:
        begin = max(idx for idx, s in enumerate(stress) if (s == 1 or s == 2))
    except ValueError:
        begin = 0
    return stress[begin], *phonemes[begin:]
