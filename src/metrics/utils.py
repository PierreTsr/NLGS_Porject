"""
    utils.py
    Created by Pierre Tessier
    10/27/22 - 11:33 PM
    Description:
    # Enter file description
 """
import numpy as np
import torch


def to_nested_list(generations) -> list[list[int]]:
    if isinstance(generations, np.ndarray):
        if generations.ndim == 1:
            generations = [list(generations)]
        elif generations.ndim == 2:
            generations = [list(x) for x in generations]
        else:
            raise ValueError("Rhyme metrics only support up to 2 dimensional inputs, "
                             "received {}-dim".format(generations.ndim))
    elif isinstance(generations, list) and not isinstance(generations[0], list):
        generations = [generations]
    if isinstance(generations[0][0], float) or isinstance(generations[0][0], torch.Tensor):
        generations = [[int(g) for g in generation] for generation in generations]
    return generations
