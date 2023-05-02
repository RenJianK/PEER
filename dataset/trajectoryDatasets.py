#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : trajectoryDatasets.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/9/3 20:43
import torch
from torch.utils.data import Dataset

from dataset.trajectoryHighD import TrajectoryState


class TrajectoryDatasets(Dataset):
    """
        This class is prepare data for model
    """

    def __init__(self, train=True):
        self.load = TrajectoryState(train)

    def __getitem__(self, index):
        # vid_now, vid_before, rid, t,
        grid, x, y, n = self.load[index]
        inputs = torch.tensor(grid, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32).cuda()
        y = torch.tensor(y, dtype=torch.float32).cuda()

        return inputs, x, y, n

    def __len__(self):
        return len(self.load)


if __name__ == '__main__':
    td = TrajectoryDatasets()
    print(td[0])
