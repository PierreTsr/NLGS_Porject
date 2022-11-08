"""
    pronunciation_tokenizer.py
    Created by Pierre Tessier
    10/19/22 - 8:14 AM
    Description:
    Create a tokenization of for both pronunciation and stress. Used for pronunciation embeddings training.
 """
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.cmu_tools import CMULinker


class PronunciationTokenizer:

    def __init__(self, linker: CMULinker, tokenizer: PreTrainedTokenizer):
        self.linker = linker
        self.punctuation = linker.punctuation

        self.vocabulary_p = linker.cmu_dictionary.phonemes + self.punctuation
        self.index_p = {p: idx for idx, p in enumerate(self.vocabulary_p)}

        self.vocabulary_s = linker.cmu_dictionary.stress + self.punctuation
        self.index_s = {s: idx for idx, s in enumerate(self.vocabulary_s)}

        self.punctuation_ids = {tokenizer.convert_tokens_to_ids(t): (self.index_p[t], self.index_s[t])
                                for t in self.punctuation[:-1]}
        self.unk_token_p = len(self.vocabulary_p) - 1
        self.unk_token_s = len(self.vocabulary_s) - 1
        self.newline_token = tokenizer.convert_tokens_to_ids("Ċ")

    def convert_sentence_pt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pronunciation = []
        stress = []
        for token_id, mask in zip(input_ids, attention_mask):
            if not mask:
                continue
            if token_id.item() in self.punctuation_ids.keys():
                p, s = self.punctuation_ids[token_id.item()]
                pronunciation.append(p)
                stress.append(s)
                continue
            try:
                p, s = self.linker.get_pronunciation(token_id)
                pronunciation += p
                stress += s
            except KeyError:
                p, s = len(self.vocabulary_p) - 1, len(self.vocabulary_s) - 1
                pronunciation.append(p)
                stress.append(s)
        return pronunciation, stress

    def convert_sentence(self, input_ids: list[int]):
        pronunciation = []
        stress = []
        for token_id in input_ids:
            if token_id in self.punctuation_ids.keys():
                continue
            try:
                p, s = self.linker.get_pronunciation(token_id)
                pronunciation += p
                stress += s
            except KeyError:
                continue
        return pronunciation, stress


    def convert_to_phonemes(self, token_ids):
        res = []
        for token_id in token_ids:
            res.append(self.vocabulary_p[token_id])
        return res

    def convert_to_stress(self, token_ids):
        res = []
        for token_id in token_ids:
            res.append(self.vocabulary_s[token_id])
        return res

    def convert_tokens_2d_pt(self, input_ids: torch.Tensor | list, attention_mask: torch.Tensor | list, **kwargs):
        if type(input_ids) == torch.Tensor and input_ids.ndim == 1:
            pronunciation, stress = self.convert_sentence_pt(input_ids, attention_mask)
            return {"pronunciation": pronunciation, "stress": stress}

        if isinstance(input_ids, list) and isinstance(input_ids[0], int):
            input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)
            pronunciation, stress = self.convert_sentence_pt(input_ids, attention_mask)
            return {"pronunciation": pronunciation, "stress": stress}

        if type(input_ids) == list or input_ids.ndim == 2:
            pronunciation = []
            stress = []
            for token_id, mask in zip(input_ids, attention_mask):
                p, s = self.convert_sentence_pt(token_id, mask)
                pronunciation.append(p)
                stress.append(s)
            return {"pronunciation": pronunciation, "stress": stress}

        else:
            raise NotImplementedError("Incorrect Tensor dimension for conversion")

    def convert_tokens_3d(self, input_ids: list[list[int]], attention_mask: list[list[int]], max_length=8, **kwargs):
        phonemes_batch = []
        stress_batch = []
        attention_batch = []
        for token_ids, attention_mask in zip(input_ids, attention_mask):
            phonemes = []
            stress = []
            attention = []
            for token_id, mask in zip(token_ids, attention_mask):
                if not mask:
                    phonemes.append([self.unk_token_p] * max_length)
                    stress.append([self.unk_token_s] * max_length)
                    attention.append([0] * max_length)
                    continue
                if token_id in self.punctuation_ids.keys():
                    p, s = self.punctuation_ids[token_id]
                    pho = [p]
                    s = [s]
                else:
                    try:
                        pho, s = self.linker.get_pronunciation(token_id)
                    except KeyError:
                        pho = s = []
                phonemes.append(pho + [self.unk_token_p] * (max_length - len(pho)) if len(pho) <= max_length else pho[:max_length])
                stress.append(s + [self.unk_token_s] * (max_length - len(s)) if len(s) <= max_length else s[:max_length])
                attention.append([1] * min(len(pho), max_length) + [0] * (max_length - min(len(pho), max_length)))
            phonemes_batch.append(phonemes)
            stress_batch.append(stress)
            attention_batch.append(attention)
        return {"pronunciation": phonemes_batch, "stress": stress_batch, "pronunciation_attention_mask": attention_batch}

    def dataset_to_files(self, dataset: Dataset,
                         target_dir: Path = Path("data/datasets/gutenberg_pronunciation_tokenized")):

        target_dir.mkdir(exist_ok=True, parents=True)
        print("Computing the pronunciation and stress representation of the dataset:")
        dataset = dataset.map(lambda data: self.convert_tokens_2d_pt(**data), batched=False, num_proc=1)

        n = len(dataset)
        print("Writing pronunciation data to file:")
        with open(target_dir/"pronunciation.txt", "w") as file:
            for line in tqdm(dataset["pronunciation"], total=n):
                line = [str(x) for x in line]
                file.write(" ".join(line) + "\n")
        print("Writing stress data to file:")
        with open(target_dir/"stress.txt", "w") as file:
            for line in tqdm(dataset["stress"], total=n):
                line = [str(x) for x in line]
                file.write(" ".join(line) + "\n")
