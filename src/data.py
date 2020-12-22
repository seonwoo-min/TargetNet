# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np
import pandas as pd

import torch


class miRNA_CTS_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for miRNA-CTS pair data """
    def __init__(self, mi_seqs, cts_r_seqs, labels):
        self.mi_seqs = mi_seqs
        self.cts_r_seqs = cts_r_seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.mi_seqs[i], self.cts_r_seqs[i], self.labels[i]


def get_dataset_from_configs(data_cfg, split_idx=None, sanity_check=False):
    """ load miRNA-CTS dataset from config files """
    table = pd.read_csv(data_cfg.path[split_idx], sep='\s+')
    if split_idx in ["train", "val"]:
        table = table[table["split"] == split_idx]
        labels = torch.from_numpy(table['label'].values.astype(np.float32)).unsqueeze(1)
    else:
        labels = torch.from_numpy(np.zeros((len(table)), np.float32)) # dummy label for code compatibility
    if sanity_check: table, labels = table[:300], labels[:300]

    mi_seqs = [torch.from_numpy(encode_RNA(seq, esa, data_cfg.with_esa, idx="mirna"))
               for seq, esa in zip(table['mirna_seq'], table['mirna_esa'])]
    cts_r_seqs = [torch.from_numpy(encode_RNA(reverse(seq), reverse(esa), data_cfg.with_esa, idx="mrna"))
                   for seq, esa in zip(table['cts_seq'], table['cts_esa'])]
    dataset = miRNA_CTS_dataset(mi_seqs, cts_r_seqs, labels)

    return dataset


def encode_RNA(seq, esa, with_esa, idx):
    """ one-hot encoder for RNA sequences with/without extended seed alignments """
    chars = {"A":0, "C":1, "G":2, "U":3}
    if idx == "mirna": length = 30
    else:              length = 40

    seq = seq.upper()
    if not with_esa:
        x = np.zeros((len(chars), length), dtype=np.float32)
        for i in range(len(seq)):
            x[chars[seq[i]], i] = 1
    else:
        chars["-"] = 4
        length += 10
        esa = esa.upper()

        x = np.zeros((len(chars), length), dtype=np.float32)
        if idx == "mirna":
            for i in range(len(esa)):
                x[chars[esa[i]], i] = 1
            for i in range(10, len(seq)):
                x[chars[seq[i]], len(esa) + i-10] = 1
        else:
            for i in range(5):
                x[chars[seq[i]], i] = 1
            for i in range(len(esa)):
                x[chars[esa[i]], 5 + i] = 1
            for i in range(15, len(seq)):
                x[chars[seq[i]], 5+len(esa) + i-15] = 1

    return x


def reverse(seq):
    """ reverse the given sequence """
    seq_r = ""
    for i in range(len(seq)):
        seq_r += seq[len(seq) - 1 - i]
    return seq_r

