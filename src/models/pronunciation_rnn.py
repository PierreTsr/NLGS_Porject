"""
    pronunciation_rnn.py
    Created by Pierre Tessier
    10/19/22 - 11:04 PM
    Description:
    # Enter file description
 """
import torch
from torch import nn


class PronunciationRNN(nn.Module):
    def __init__(self, embeddings_p: torch.Tensor, embeddings_s: torch.Tensor, dim_target: int, layers: int = 2,
                 rnn: str = "gru"):
        super(PronunciationRNN, self).__init__()
        dim_p, dim_s = embeddings_p.size()[-1], embeddings_s.size()[-1]
        self.dim_target = dim_target
        self.embeddings_p = nn.Embedding.from_pretrained(embeddings_p, False)
        self.embeddings_s = nn.Embedding.from_pretrained(embeddings_s, False)
        if rnn == "lstm":
            self.rnn = nn.LSTM(dim_p + dim_s, dim_target // 2, layers, bidirectional=True, batch_first=True)
        elif rnn == "gru":
            self.rnn = nn.GRU(dim_p + dim_s, dim_target // 2, layers, bidirectional=True, batch_first=True)
        else:
            raise ValueError("rnn can only take values in {\"lstm\", \"gru\"}")

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
        x, _ = self.rnn(x)  # (B*L)xWxT
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # (B*L)xWxT
        # x = torch.sum(x, dim=1) / lengths[mask].view(-1, 1).expand(-1, x.size()[2])  # (B*L)xT

        res = torch.zeros((lengths.size()[0], self.dim_target), device=device)
        res = torch.index_copy(res, 0, indices, x)
        res = res.reshape((*init_size[:2], self.dim_target))

        return res


class PronunciationAttention(nn.Module):

    def __init__(self, embeddings_p: torch.Tensor, embeddings_s: torch.Tensor, dim_target: int, dim_hidden: int = 128,
                 max_length: int = 8, num_heads: int = 4):
        super(PronunciationAttention, self).__init__()
        dim_p, dim_s = embeddings_p.size()[-1], embeddings_s.size()[-1]
        self.dim_src = dim_p + dim_s
        self.dim_target = dim_target
        self.embeddings_p = nn.Embedding.from_pretrained(embeddings_p, False)
        self.embeddings_s = nn.Embedding.from_pretrained(embeddings_s, False)
        self.positional_embeddings = nn.Embedding(max_length, self.dim_src)
        self.attention = nn.MultiheadAttention(self.dim_src, num_heads, batch_first=True)
        self.bn1 = nn.BatchNorm1d(self.dim_src)
        self.fc1 = nn.Linear(self.dim_src, dim_hidden)
        self.activation = nn.PReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_target)
        self.bn2 = nn.BatchNorm1d(dim_target, affine=False)

    def forward(self,
                pronunciation: torch.Tensor,
                stress: torch.Tensor,
                pronunciation_attention_mask: torch.Tensor,
                **kwargs):
        init_size = pronunciation_attention_mask.size()  # BxLxW
        device = pronunciation.device
        pronunciation, stress, syllable_mask = torch.flatten(pronunciation, end_dim=1), \
                                               torch.flatten(stress, end_dim=1), \
                                               torch.flatten(pronunciation_attention_mask,
                                                             end_dim=1)  # (B*L)xW

        p = self.embeddings_p(pronunciation.int())  # (B*L)xWxP
        s = self.embeddings_s(stress.int())  # (B*L)xWxS
        x = torch.concat([p, s], dim=-1)
        print(p.dtype, s.dtype, x.dtype)

        word_mask = torch.any(syllable_mask, dim=1)
        lengths = torch.sum(syllable_mask, dim=1)[word_mask]
        syllable_mask = syllable_mask[word_mask]
        res_size = (*x.shape[:-2], self.dim_target)
        indices = torch.arange(0, x.shape[0], device=device, dtype=torch.long)[word_mask]

        x = x[word_mask]
        position = torch.arange(0, x.shape[1], device=device, dtype=torch.int).expand(x.shape[0], x.shape[1])
        w = self.positional_embeddings(position)
        x += w
        x = x * syllable_mask.float().view(*syllable_mask.shape, 1).expand(*x.shape)
        print(w.dtype, x.dtype)
        print([p.dtype for p in self.attention.parameters()])

        y, _ = self.attention(x, x, x, key_padding_mask=torch.logical_not(syllable_mask), need_weights=False)
        x += y * syllable_mask.float().view(*syllable_mask.shape, 1).expand(*x.shape)
        x = torch.sum(x, dim=1) / lengths.view(-1, 1).expand(-1, x.shape[-1])

        x = self.bn1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.bn2(x)

        res = torch.zeros(res_size, dtype=x.dtype, device=device)
        res.index_copy_(0, indices, x)
        res = res.reshape((*init_size[:2], self.dim_target))

        return res
