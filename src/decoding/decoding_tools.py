"""
    decoding_tools.py
    Created by Pierre Tessier
    11/10/22 - 6:20 PM
    Description:
    # Enter file description
 """
from abc import ABC, abstractmethod

import torch
from transformers import LogitsProcessor, StoppingCriteria

from src.utils import to_nested_list
from .a_star_utils import generate_samples


class MeterLogitsProcessor(LogitsProcessor):

    def __init__(self, meter_fn, mixin_coeff: float):
        self.meter_fn = meter_fn
        self.mixin = mixin_coeff

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generations = to_nested_list(input_ids)
        meter_scores = torch.zeros_like(scores)
        for i, gen in enumerate(generations):
            meter_scores[i, :] = self.meter_fn(gen)
        return scores + self.mixin * meter_scores


class VerseCountStoppingCriteria(StoppingCriteria):

    def __init__(self, n_verses, new_line_token_id=198):
        self.n = n_verses
        self.token_id = new_line_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        mask = input_ids == self.token_id
        new_lines = (mask[:, 1:] ^ mask.roll(1, dims=1)[:, 1:]) & mask[:, 1:]
        count = torch.max(torch.sum(new_lines, dim=1)).item()
        return count >= self.n


class AuxLogitsProcessor(ABC):

    @abstractmethod
    def __call__(self, samples: torch.LongTensor, voc_idx: torch.LongTensor, batch_idx: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        ...


class RhymeAuxLogitsProcessor(AuxLogitsProcessor):

    def __init__(self, rhyme_fn, mixin_coeff: float):
        self.rhyme_fn = rhyme_fn
        self.mixin = mixin_coeff

    def __call__(self, samples: torch.LongTensor, voc_idx: torch.LongTensor, batch_idx: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        rhyme_scores = torch.zeros_like(scores, device="cpu")
        samples = [to_nested_list(sample) for sample in samples]
        for gen, word, batch in zip(samples, voc_idx, batch_idx):
            rhyme_scores[batch, word] = self.rhyme_fn(gen)
        rhyme_scores[batch_idx, voc_idx] -= torch.mean(rhyme_scores[batch_idx, voc_idx])
        return scores + self.mixin * rhyme_scores.to(scores.device)


class AlliterationAuxLogitsProcessor(AuxLogitsProcessor):

    def __init__(self, alliteration_fn, mixin_coeff: float):
        self.alliteration_fn = alliteration_fn
        self.mixin = mixin_coeff

    def __call__(self, samples: torch.LongTensor, voc_idx: torch.LongTensor, batch_idx: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        alliteration_scores = torch.zeros_like(scores, device="cpu")
        samples = [to_nested_list(sample) for sample in samples]
        for gen, word, batch in zip(samples, voc_idx, batch_idx):
            alliteration_scores[batch, word] = self.alliteration_fn(gen)
        alliteration_scores[batch_idx, voc_idx] -= torch.mean(alliteration_scores[batch_idx, voc_idx])
        return scores + self.mixin * alliteration_scores.to(scores.device)


class AStarLogitsProcessor(LogitsProcessor):

    def __init__(self, model, aux_processors: list[AuxLogitsProcessor], top_k: int, max_new_tokens: int, **kwargs):
        self.model = model
        self.eos_token_id = 50256
        self.newline_token_id = 198
        self.aux_processors = aux_processors
        self.k = top_k
        self.max_new_tokens = max_new_tokens
        self.kwargs = kwargs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, voc_size = scores.shape
        num_beams = self.kwargs["num_beams"] if "num_beams" in self.kwargs.keys() else 1
        topk = torch.topk(scores, self.k, dim=-1).indices
        samples = torch.cat((input_ids.view(batch_size, 1, -1).repeat(1, self.k, 1), topk.view(batch_size, self.k, 1)), dim=-1).long()
        samples = samples.view(batch_size * self.k, -1)
        samples = generate_samples(self.model, samples, self.eos_token_id, self.max_new_tokens,
                                   newline_token_id=self.newline_token_id, **self.kwargs)
        samples = samples.cpu()

        samples = samples.view(batch_size * self.k, num_beams, -1)
        batch_idx = torch.arange(0, batch_size, device="cpu").view(batch_size, 1).repeat(1, self.k).flatten()
        voc_idx = topk.flatten().cpu()
        for aux in self.aux_processors:
            scores = aux(samples, voc_idx, batch_idx, scores)
        return scores
