#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : dataset.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/5/16 15:40
import torch
from torch.utils.data import Dataset
from dataset.loadData import LoadData
from torchvision.transforms import Normalize


# ---- generate data with tensor format ----


class RepaintDatasets(Dataset):
    """
        This class is prepare data for model
    """

    def __init__(self, train=True):
        self.load = LoadData(train)
        self.normal = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # [len(self.load.input) * 7 // 8:]
        # [:len(self.load.input) // 8]
        # if train:
        #     self.input = torch.tensor(self.load.input, dtype=torch.float32)
        #     self.real = torch.tensor(self.load.real, dtype=torch.float32)
        # else:
        #     self.input = torch.tensor(self.load.input, dtype=torch.float32)
        #     self.real = torch.tensor(self.load.real, dtype=torch.float32)
        # print(self.input.shape)
        # self.input = self.normal(self.input)
        # print(self.input.shape)

    def __getitem__(self, index):
        inputs, reals = self.load[index]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        reals = torch.tensor(reals, dtype=torch.float32)
        # return self.input[index], self.real[index][:, :, -2:]
        return inputs, reals[:, :, -1]

    def __len__(self):
        return len(self.load)


if __name__ == '__main__':
    dataset = RepaintDatasets(train=True)
    print(len(dataset))
    inp, real = dataset[0]
    print(inp.shape)
    print(real.shape)
