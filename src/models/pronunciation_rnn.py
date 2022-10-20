"""
    pronunciation_rnn.py
    Created by Pierre Tessier
    10/19/22 - 11:04 PM
    Description:
    # Enter file description
 """
import torch
from torch import nn


class PronunciationLSTM(nn.Module):
    def __init__(self, embeddings_p: torch.Tensor, embeddings_s: torch.Tensor, dim_target: int):
        super().__init__()
        dim_p, dim_s = embeddings_p.size()[-1], embeddings_s.size()[-1]
        self.dim_target = dim_target
        self.embeddings_p = nn.Embedding.from_pretrained(embeddings_p, False)
        self.embeddings_s = nn.Embedding.from_pretrained(embeddings_s, False)
        self.lstm = nn.LSTM(dim_p + dim_s, dim_target // 2, 2, bidirectional=True, batch_first=True)

    def forward(self,
                pronunciation: torch.Tensor,
                stress: torch.Tensor,
                pronunciation_attention_mask: torch.Tensor,
                **kwargs):
        init_size = pronunciation_attention_mask.size()  # BxLxW
        device = pronunciation.device
        pronunciation, stress, pronunciation_attention_mask = torch.flatten(pronunciation, end_dim=1), \
                                                              torch.flatten(stress, end_dim=1), \
                                                              torch.flatten(pronunciation_attention_mask,
                                                                            end_dim=1)  # (B*L)xW

        lengths = torch.sum(pronunciation_attention_mask, dim=-1)  # (B*L)
        mask = lengths > 0
        indices = torch.arange(0, lengths.size()[0], dtype=torch.long, device=device)[mask]

        pronunciation, stress = pronunciation[mask], stress[mask]
        p = self.embeddings_p(pronunciation.int())  # (B*L)xWxP
        s = self.embeddings_s(stress.int())  # (B*L)xWxS

        x = torch.concat([p, s], dim=-1)  # (B*L)xWx(S+P)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths[mask].cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)  # (B*L)xWxT
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # (B*L)xWxT
        x = torch.sum(x, dim=1) / lengths[mask].view(-1, 1).expand(-1, x.size()[2])  # (B*L)xT

        res = torch.zeros((lengths.size()[0], self.dim_target), device=device)
        res = torch.index_copy(res, 0, indices, x)
        res = res.reshape((*init_size[:2], self.dim_target))

        return res
