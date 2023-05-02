#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : GroupPredictModel.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/8/26 22:44

import torch
from torch import nn, relu, softmax


class PredictGenerator(nn.Module):
    """Summary of class here.

    The Input Shape is [5, 8, 10, 12] [s-length, channel, 10, 12]
    Longer class information....

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, input_channel):
        super().__init__()

        self.s_3d = nn.Sequential(
            nn.Conv3d(
                in_channels=input_channel, out_channels=input_channel * 4,
                kernel_size=(2, 4, 3), padding=(0, 0, 2), stride=(1, 2, 1)
            ),
            nn.BatchNorm3d(input_channel * 4),
            nn.ReLU(True),  # [3, 19, 5]
            nn.Conv3d(
                in_channels=input_channel * 4, out_channels=input_channel * 16,
                kernel_size=(3, 3, 3), padding=(1, 0, 1), stride=(1, 2, 1)
            ),
            nn.BatchNorm3d(input_channel * 16),
            nn.ReLU(True),  # [3, 9, 5]
            nn.Conv3d(
                in_channels=input_channel * 16, out_channels=input_channel * 16,
                kernel_size=(3, 3, 3), padding=(1, 0, 1)
            ),
            nn.BatchNorm3d(input_channel * 16),
            nn.ReLU(True),  # [3, 7, 5]
            nn.Conv3d(
                in_channels=input_channel * 16, out_channels=input_channel * 16,
                kernel_size=(3, 3, 3), padding=(1, 0, 1)
            ),
            nn.BatchNorm3d(input_channel * 16),
            nn.ReLU(True),
        )  # [3, 5, 5]

        self.c_3d = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=input_channel * 16, out_channels=input_channel * 16,
                        kernel_size=(3, 3, 3)
                        , padding=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(input_channel * 16),
                    nn.ReLU(True),
                    nn.Conv3d(
                        in_channels=input_channel * 16, out_channels=input_channel * 16,
                        kernel_size=(3, 3, 3)
                        , padding=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(input_channel * 16),
                ) for _ in range(1)
            ]
        )

        self.e_3d = nn.Sequential(
            nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(input_channel * 16),  # [3, 3, 3]
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(input_channel * 16),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(input_channel * 16),
            nn.ReLU(True),
        )

        # self.produce_3d = nn.Sequential(
        #     nn.Conv3d(in_channels=input_channel, out_channels=input_channel * 2, kernel_size=(2, 3, 3),
        #               padding=(1, 1, 0)),  # []
        #     nn.BatchNorm3d(input_channel * 2),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 2, out_channels=input_channel * 4, kernel_size=(3, 3, 3),
        #               padding=(0, 1, 1)),
        #     nn.BatchNorm3d(input_channel * 4),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 4, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
        #               padding=(1, 1, 1)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
        #               padding=(1, 1, 1)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
        #               padding=(1, 1, 1)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
        #               padding=(1, 1, 1)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
        #               padding=(1, 0, 0)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 4, 4),
        #               padding=(1, 0, 0)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
        #               padding=(1, 0, 0)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel * 16, out_channels=input_channel * 16, kernel_size=(3, 3, 3),
        #               padding=(0, 0, 0)),
        #     nn.BatchNorm3d(input_channel * 16),
        #     nn.ReLU(True),
        # )

        self.flatten = nn.Sequential(
            nn.Linear(input_channel * 16, 300),
            nn.ReLU(True),
        )

        self.residual = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(300, 400),
                nn.ReLU(True),
                nn.Linear(400, 300)
            ) for _ in range(1)]
        )

        self.predict = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 120),
            # nn.ReLU()
        )

        self.line = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 120),
            # nn.ReLU()
        )

    def forward(self, x):
        """

        Args:
            x: shape [N, Channel, Sequence-length, height, width]

        Returns:

        """
        o = self.s_3d(x)
        for i in range(len(self.c_3d)):
            o_o = self.c_3d[i](o)
            o = relu(o + o_o)

        # print(o.shape)
        o = self.e_3d(o)

        # o = self.produce_3d(x)  # [N, channel * n, 1, 1, 1]
        o = torch.reshape(o, (x.shape[0], -1))
        y = self.flatten(o)

        for i in range(len(self.residual)):
            o = self.residual[i](y)
            y = relu(y + o)

        t = y
        l = self.line(y)
        y = self.predict(y)

        return y, l, t
