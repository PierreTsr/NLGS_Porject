"""
    a_star_utils.py
    Created by Pierre Tessier
    11/14/22 - 2:45 PM
    Description:
    # Enter file description
 """
import torch
import torch.nn.functional as F
from tqdm import tqdm


def generate_samples(model, input_ids: torch.LongTensor, eos_token_id: int, max_new_tokens: int, **kwargs):
    batch_size = input_ids.shape[0]
    if "num_beams" not in kwargs.keys():
        num_beams = 1
        generations = input_ids.clone().detach().view(batch_size, 1, -1)
    else:
        num_beams = kwargs["num_beams"]
        generations = input_ids.clone().detach().view(batch_size, 1, -1).repeat(1, kwargs["num_beams"], 1)

    if "sample" in kwargs.keys():
        sample = kwargs["sample"]
    else:
        sample = False

    if "temperature" in kwargs.keys():
        temperature = kwargs["temperature"]
    else:
        temperature = 1.0

    if "n_newlines" in kwargs.keys():
        n_newlines = kwargs["n_newlines"]
        newline_token_id = kwargs["newline_token_id"]
    else:
        n_newlines = 0
        newline_token_id = None

    first = True
    start_idx = generations.shape[-1]
    active = (generations[..., -1] != eos_token_id).flatten()
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens), desc="A-star sample generator", leave=False):
            logits = model(input_ids=generations.view(batch_size * num_beams, -1), use_cache=True).logits
            logits = logits[:, -1, :] / temperature

            if sample:
                samples = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                samples[torch.logical_not(active)] = eos_token_id
                generations = torch.cat((generations, samples.view(batch_size, num_beams, 1)), dim=-1)
            else:
                voc_size = logits.shape[-1]
                pos = torch.nonzero(torch.logical_not(active))
                logits.view(-1, voc_size)[pos[:, 0], :] = -torch.inf
                logits.view(-1, voc_size)[pos[:, 0], torch.zeros_like(pos[:, 0]) + eos_token_id] = torch.inf
                if first:
                    indices_voc = torch.arange(0, voc_size)
                    indices_seq = torch.zeros_like(indices_voc)
                    top_k = torch.topk(logits.view(batch_size, num_beams, -1)[:, 0, :], num_beams, dim=-1).indices
                    first = False
                else:
                    indices_voc = torch.arange(0, voc_size).view(1, -1).repeat(num_beams, 1).view(-1)
                    indices_seq = torch.arange(0, num_beams).view(-1, 1).repeat(1, voc_size).view(-1)
                    top_k = torch.topk(logits.view(batch_size, -1), num_beams, dim=-1).indices
                top_k_voc = indices_voc[top_k.flatten()].view(batch_size, num_beams, 1)
                top_k_seq = indices_seq[top_k.flatten()].view(batch_size, num_beams)
                batch_idx = torch.arange(0, batch_size).view(-1, 1).expand(-1, num_beams)
                generations = torch.cat((generations[batch_idx, top_k_seq, :], top_k_voc), dim=-1)

            if n_newlines:
                mask = (generations[..., start_idx:-1] == newline_token_id)
                newlines = (mask[..., 1:] ^ mask.roll(1, dims=-1)[..., 1:]) & mask[..., 1:]
                done = torch.nonzero(torch.sum(newlines, dim=-1) >= n_newlines)
                generations[done[:, 0], done[:, 1], torch.zeros(done.shape[0], dtype=torch.long) - 1] = eos_token_id

            active = (generations[..., -1] != eos_token_id).flatten()
            if not torch.any(active):
                break
    return generations
