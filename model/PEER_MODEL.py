#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : PEER_MODEL.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/9/18 21:49


import torch
from torch import nn, relu
import torch.nn.functional as F

# input shape is (5, 67, 3, 8)
# output shape is (output_window, 3, 4)

# (57, 67, 3, 8) -> generator: conv3D -> ConvGRU -> Transport Conv -> (output_window, 3, 4)
# [(57, 67, 3, 8), (output_window, 3, 4)] -> discriminator ->
# conv3D -> ConvGRU, conv2D -> concatenation -> FC -> true or false
# Loss function: L2 + L gan
from tool import cnn_paras_count


class Conv3D_G(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # 3d conv + residual
        self.s = nn.Sequential(
            nn.Conv3d(in_channels=input_channel, out_channels=input_channel * 2, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.BatchNorm3d(input_channel * 2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * 2, out_channels=input_channel * 4, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.BatchNorm3d(input_channel * 4),
            nn.ReLU(True),
        )

        self.c = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(in_channels=input_channel * 4, out_channels=input_channel * 4, padding=(1, 1, 1),
                                     kernel_size=(3, 3, 3),
                                     stride=1)
                           , nn.BatchNorm3d(input_channel * 4)
                           ) for _ in range(8)])

        self.e = nn.Sequential(
            nn.Conv3d(in_channels=input_channel * 4, out_channels=input_channel * 2, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.BatchNorm3d(input_channel * 2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * 2, out_channels=input_channel, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.BatchNorm3d(input_channel),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel, out_channels=output_channel,
                      padding=(1, 1, 1),
                      # padding=(0, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        o = self.s(x)

        for i in range(len(self.c)):
            y = self.c[i](o)
            o = relu(o + y)

        o = self.e(o)
        return o


class Conv3D_D(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # 3d conv + residual
        self.s = nn.Sequential(
            nn.Conv3d(in_channels=input_channel, out_channels=input_channel * 2, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.LayerNorm([40, 3]),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * 2, out_channels=input_channel * 4, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.LayerNorm([40, 3]),
            nn.ReLU(True),
        )

        self.c = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(in_channels=input_channel * 4, out_channels=input_channel * 4, padding=(1, 1, 1),
                                     kernel_size=(3, 3, 3),
                                     stride=1)
                           , nn.LayerNorm([40, 3])
                           ) for _ in range(8)])

        self.e = nn.Sequential(
            nn.Conv3d(in_channels=input_channel * 4, out_channels=input_channel * 2, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.LayerNorm([40, 3]),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * 2, out_channels=input_channel, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.LayerNorm([40, 3]),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel, out_channels=output_channel,
                      padding=(1, 1, 1),
                      # padding=(0, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.LayerNorm([40, 3]),
            nn.ReLU(True),
        )

    def forward(self, x):
        o = self.s(x)

        for i in range(len(self.c)):
            y = self.c[i](o)
            o = relu(o + y)

        o = self.e(o)
        return o


class ConvGRUCell(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.conv_x_z = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1)
        self.conv_h_z = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1)

        self.conv_x_r = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1)
        self.conv_h_r = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1)
        self.conv_u = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1)

        self.conv_out = nn.Conv2d(in_channels=hidden_channel, out_channels=out_channel, kernel_size=1)

    def forward(self, x, h):
        z_t = torch.sigmoid((self.conv_x_z(x) + self.conv_h_z(h)))
        r_t = torch.sigmoid((self.conv_x_r(x) + self.conv_h_r(h)))

        h_hat_t = torch.tanh((self.conv(x) + self.conv_u(torch.mul(r_t, h))))
        h_t = torch.mul((1 - z_t), h) + torch.mul(z_t, h_hat_t)

        y = self.conv_out(h_t)

        return y, h_t


class ConvGRULayer(nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel):
        super().__init__()
        self.hidden_size = hidden_channel
        self.out_channel = output_channel
        self.gru_cell = ConvGRUCell(input_channel, hidden_channel, output_channel)
        self.gru = torch.nn.GRU

    def forward(self, x, h_0=None):
        batch, seq_len, _, _, _ = x.size()

        hidden_seq = torch.zeros((seq_len, batch, self.hidden_size, x.shape[-2], x.shape[-1])).cuda()
        o = torch.zeros((seq_len, batch, self.out_channel, x.shape[-2], x.shape[-1])).cuda()
        if h_0 is None:
            h_t = torch.zeros((batch, self.hidden_size, x.shape[-2], x.shape[-1])).cuda()
        else:
            h_t = h_0

        for t in range(seq_len):
            i = torch.permute(x, (1, 0, 2, 3, 4))[t]
            y, h_t = self.gru_cell(i, h_t)
            o[t] = y
            hidden_seq[t] = h_t

        hidden_seq = torch.permute(hidden_seq, (1, 0, 2, 3, 4))
        o = torch.permute(o, (1, 0, 2, 3, 4))
        return o, hidden_seq


class Conv2D(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        print(input_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=2 * input_channel, kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(2 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=2 * input_channel, out_channels=4 * input_channel, kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(2 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * input_channel, out_channels=8 * input_channel, kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(4 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8 * input_channel, out_channels=16 * input_channel, kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(8 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * input_channel, out_channels=4 * input_channel, kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(16 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * input_channel, out_channels=input_channel, kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(4 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
        )

    def forward(self, x):
        y = torch.zeros((x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4])).cuda()
        x = torch.permute(x, (1, 0, 2, 3, 4))
        for i in range(x.shape[0]):
            y[:, i] = self.conv(x[i])

        return y


class Generator(nn.Module):
    """Summary of class here.

    Longer class information....

    Args:
        sequence_length (int): Length of time sequence.
        input_channel (int): Number of channels in the input state.
        input_height (int): Height of input, i.e. the number of gird along road.
        output_channel (int): Number of channels in the output state.

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(batch, t, 8, 40, 3)`

    Outputs: o
        * **output**: tensor of shape :math:`(batch, output_channel, output_window, 3)`
    """

    def __init__(self, sequence_length, input_channel, input_height, output_channel):
        super().__init__()
        self.conv3d = Conv3D_G(input_channel, input_channel)
        # [batch, t, 4, 67, 3]
        self.conv_gru = ConvGRULayer(input_channel, input_channel, 1)
        self.conv2d_s = nn.Sequential(
            nn.Conv2d(in_channels=sequence_length, out_channels=2 * sequence_length, kernel_size=(6, 3),
                      padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(2 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=2 * sequence_length, out_channels=4 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(4 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * sequence_length, out_channels=8 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(8 * sequence_length),
            nn.ReLU(True),
        )
        self.conv2d_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=8 * sequence_length, out_channels=8 * sequence_length, kernel_size=3, padding=1),
                nn.BatchNorm2d(8 * sequence_length),
            )
            for _ in range(8)
        ])

        self.conv2d_e = nn.Sequential(
            nn.Conv2d(in_channels=8 * sequence_length, out_channels=16 * sequence_length, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * sequence_length, out_channels=32 * sequence_length, kernel_size=3),
            nn.BatchNorm2d(32 * sequence_length),
            nn.ReLU(True)
        )

        self.transCNN = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * sequence_length, out_channels=output_channel * 8, kernel_size=4,
                               stride=2,
                               padding=1, bias=False),  # [2, 2]
            nn.BatchNorm2d(output_channel * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel * 8, out_channels=output_channel * 4, kernel_size=4, stride=1,
                               padding=1, bias=False),  # [3, 3]
            nn.BatchNorm2d(output_channel * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel * 4, out_channels=output_channel, kernel_size=(3, 3),
                               stride=(2, 1), padding=(1, 1), bias=False),  # [5, 3]
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            # nn.ConvTranspose2d(in_channels=output_channel * 2, out_channels=output_channel, kernel_size=(4, 3),
            #                    stride=(2, 1), padding=(2, 1), bias=False),
            # nn.ReLU(True)
        )

        self.seg = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=4 * sequence_length, kernel_size=3, padding=(0, 1)),
            nn.BatchNorm2d(4 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * sequence_length, out_channels=8 * sequence_length, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8 * sequence_length, out_channels=16 * sequence_length, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * sequence_length, out_channels=32 * sequence_length, kernel_size=3),
            nn.BatchNorm2d(32 * sequence_length),
            nn.ReLU(True),
        )

        self.group = nn.Sequential(  # [batch, channel, 40, 3]
            nn.Conv2d(in_channels=3, out_channels=4 * sequence_length, kernel_size=(4, 3), padding=(0, 1),
                      stride=(2, 1)),
            nn.BatchNorm2d(4 * sequence_length),  # [.., .., 19, 3]
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * sequence_length, out_channels=8 * sequence_length, kernel_size=3, padding=(0, 1),
                      stride=(2, 1)),  # [.., .., 9, 3]
            nn.BatchNorm2d(8 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8 * sequence_length, out_channels=16 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)),
            nn.BatchNorm2d(16 * sequence_length),  # [.., .., 6, 3]
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * sequence_length, out_channels=16 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)),
            nn.BatchNorm2d(16 * sequence_length),  # [.., .., 3, 3]
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * sequence_length, out_channels=32 * sequence_length, kernel_size=3),
            nn.BatchNorm2d(32 * sequence_length),
            nn.ReLU(True),
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(32 * sequence_length, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 32 * sequence_length),
            nn.ReLU()
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(32 * sequence_length * 2, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 32 * sequence_length),
            nn.ReLU()
        )

        self.conv = Conv2D(input_channel, 1)

    def forward(
            self,
            x,
            seg_x,
            group_predicted_x,
            # global_h,
            # local_h=None
    ):
        o = self.conv3d(x)
        o = torch.permute(o, (0, 2, 1, 3, 4))
        o, h = self.conv_gru(o)
        o = torch.reshape(o, (o.shape[0], o.shape[1], o.shape[3], o.shape[4]))
        o = self.conv2d_s(o)

        for i in range(len(self.conv2d_c)):
            y = self.conv2d_c[i](o)
            o = relu(o + y)

        o = self.conv2d_e(o)
        o_seg_x = self.seg(seg_x)
        o_group_x = self.group(group_predicted_x)

        o = o + o_seg_x + o_group_x
        # o = o + o_seg_x

        # o = o.reshape((o.shape[0], o.shape[1]))

        # o_g = self.fc_1(global_h)
        # o = o + o_g
        # o = torch.cat((o, o_g), dim=-1)
        # o = self.fc_2(o)
        # o_1 = o.reshape((o.shape[0], o.shape[1], 1, 1))
        # o = torch.permute(o, (0, 2, 1, 3))
        y = self.transCNN(o)

        return o, y


class Global_Generator(nn.Module):
    def __init__(self, sequence_length, input_channel, input_height, output_channel):
        super().__init__()
        self.conv3d = Conv3D_G(input_channel, 1)
        self.conv2d_s_s = nn.Sequential(
            nn.Conv2d(in_channels=sequence_length, out_channels=2 * sequence_length, kernel_size=(6, 3),
                      padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(2 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=2 * sequence_length, out_channels=4 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(4 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * sequence_length, out_channels=8 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(8 * sequence_length),
            nn.ReLU(True),
        )
        self.conv2d_s_c = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=8 * sequence_length, out_channels=8 * sequence_length, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * sequence_length))
            for _ in range(8)
        ])
        self.conv2d_s_e = nn.Sequential(
            nn.Conv2d(in_channels=8 * sequence_length, out_channels=16 * sequence_length, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * sequence_length, out_channels=32 * sequence_length, kernel_size=3),
            nn.BatchNorm2d(32 * sequence_length),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * sequence_length, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 32 * sequence_length),
            nn.ReLU(True)
        )

        self.conv2d_o_s = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=2 * input_channel, kernel_size=(6, 3),
                      padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(2 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=2 * input_channel, out_channels=4 * input_channel, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(4 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * input_channel, out_channels=8 * input_channel, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(8 * input_channel),
            nn.ReLU(True)
        )
        self.conv2d_o_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=8 * input_channel, out_channels=8 * input_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(8 * input_channel))
            for _ in range(8)
        ])
        self.conv2d_o_e = nn.Sequential(
            nn.Conv2d(in_channels=8 * input_channel, out_channels=16 * input_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * input_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * input_channel, out_channels=20 * input_channel, kernel_size=3),
            nn.BatchNorm2d(20 * input_channel),
            nn.ReLU(True)
        )

        self.transCNN = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * sequence_length, out_channels=output_channel * 16, kernel_size=4,
                               stride=2,
                               padding=1, bias=False),  # [2, 2]
            nn.BatchNorm2d(output_channel * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel * 16, out_channels=output_channel * 8, kernel_size=4,
                               stride=1,
                               padding=1, bias=False),  # [3, 3]
            nn.BatchNorm2d(output_channel * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel * 8, out_channels=output_channel * 8, kernel_size=(3, 3),
                               stride=(2, 1), padding=(1, 1), bias=False),  # [5, 3]
            nn.BatchNorm2d(output_channel * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel * 8, out_channels=output_channel * 8, kernel_size=(4, 3),
                               stride=(2, 1), padding=(0, 1), bias=False),  # [12, 3]
            nn.BatchNorm2d(output_channel * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel * 8, out_channels=output_channel * 4, kernel_size=(4, 3),
                               stride=(2, 1), padding=(2, 1), bias=False),  # [22, 3]
            nn.BatchNorm2d(output_channel * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel * 4, out_channels=output_channel, kernel_size=(4, 3),
                               stride=(2, 1), padding=(3, 1), bias=False),  # [40, 3]
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False),  # [40, 3]
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
        )

    def forward(
            self,
            seq_x,  # [5, 8, 40, 3]
            # x,  # [8, 40, 3]
    ):
        o = self.conv3d(seq_x)  # [5, 40, 3]
        o = torch.reshape(o, (o.shape[0], o.shape[2], o.shape[3], o.shape[4]))
        o = self.conv2d_s_s(o)  # [160, 1, 1]

        for i in range(len(self.conv2d_s_c)):
            y = self.conv2d_s_c[i](o)
            o = relu(o + y)

        o = self.conv2d_s_e(o)

        o = o.reshape((o.shape[0], o.shape[1]))
        o = self.fc(o)
        o_1 = o.reshape((o.shape[0], o.shape[1], 1, 1))

        # o_o = self.conv2d_o_s(x)  # [160, 1, 1]
        #
        # for i in range(len(self.conv2d_o_c)):
        #     y = self.conv2d_o_c[i](o_o)
        #     o_o = relu(o_o + y)
        #
        # o_o = self.conv2d_o_e(o_o)

        y = self.transCNN(o_1)  # [2, 40, 3]

        return o, y


class Discriminator(nn.Module):
    def __init__(self, sequence_length, input_channel, input_height, output_channel):
        super().__init__()
        self.conv3d = Conv3D_D(input_channel, input_channel)
        self.conv_gru = ConvGRULayer(input_channel, input_channel, 1)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=sequence_length, out_channels=2 * sequence_length, kernel_size=3, padding=1),
            nn.LayerNorm([2 * sequence_length, input_height, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=2 * sequence_length, out_channels=4 * sequence_length, kernel_size=3, padding=1),
            nn.LayerNorm([4 * sequence_length, input_height, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=4 * sequence_length, out_channels=sequence_length, kernel_size=3, padding=1),
            nn.LayerNorm([sequence_length, input_height, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=sequence_length, out_channels=1, kernel_size=3),
            nn.LayerNorm([1, input_height - 2, 3 - 2]),
            nn.LeakyReLU(0.2, True),
        )
        # [batch, 2, 10, 3]
        self.res = nn.Sequential(
            # nn.Conv2d(in_channels=output_channel, out_channels=output_channel * 4, kernel_size=3, padding=1),
            # nn.ReLU(True),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel * 8, kernel_size=3, padding=1),
            nn.LayerNorm([output_channel * 8, 5, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=output_channel * 8, out_channels=output_channel * 16, kernel_size=3, padding=1,
                      stride=(2, 1)),
            nn.LayerNorm([output_channel * 16, 3, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=output_channel * 16, out_channels=output_channel * 32, kernel_size=3),
            nn.Flatten(),
            nn.Linear(output_channel * 32, 64),
            nn.LeakyReLU(0.2, True),
        )

        self.fc = nn.Sequential(
            nn.Linear(input_height + 64 - 2, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

        self.group = nn.Sequential(  # [batch, channel, 40, 3]
            nn.Conv2d(in_channels=3, out_channels=4 * output_channel, kernel_size=(4, 3), padding=(0, 1),
                      stride=(2, 1)),
            nn.BatchNorm2d(4 * output_channel),  # [.., .., 19, 3]
            nn.ReLU(True),
            nn.Conv2d(in_channels=4 * output_channel, out_channels=8 * output_channel, kernel_size=3, padding=(0, 1),
                      stride=(2, 1)),  # [.., .., 9, 3]
            nn.BatchNorm2d(8 * output_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8 * output_channel, out_channels=16 * output_channel, kernel_size=(4, 3),
                      padding=(0, 1)),
            nn.BatchNorm2d(16 * output_channel),  # [.., .., 6, 3]
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * output_channel, out_channels=16 * output_channel, kernel_size=(4, 3),
                      padding=(0, 1)),
            nn.BatchNorm2d(16 * output_channel),  # [.., .., 3, 3]
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * output_channel, out_channels=32 * output_channel, kernel_size=3),
            nn.BatchNorm2d(32 * output_channel),
            nn.ReLU(True),
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(32 * sequence_length, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 64),
            nn.ReLU()
        )

    def forward(self, x, y
                , group_predicted_x
                # , global_h
                ):
        o = self.conv3d(x)
        o = torch.permute(o, (0, 2, 1, 3, 4))
        o_y = self.res(y)

        o, h = self.conv_gru(o)
        o = torch.reshape(o, (o.shape[0], o.shape[1], o.shape[3], o.shape[4]))
        o = self.conv2d(o)
        o = torch.reshape(o, (o.shape[0], o.shape[2]))
        o_group_x = self.group(group_predicted_x)
        o_group_x = torch.reshape(o_group_x, (o_group_x.shape[0], o_group_x.shape[1]))
        # o_1 = torch.cat((global_h, local_h), dim=1)
        # o_1 = self.fc_1(global_h)
        o = torch.cat((o, o_group_x + o_y), dim=-1)
        # o = torch.cat((o, o_y), dim=-1)

        y = self.fc(o)
        return y


class Global_Discriminator(nn.Module):
    def __init__(self, sequence_length, input_channel, input_height, output_channel):
        super().__init__()
        self.conv3d = Conv3D_D(input_channel, 1)
        self.conv2d_s = nn.Sequential(
            nn.Conv2d(in_channels=sequence_length, out_channels=2 * sequence_length, kernel_size=(6, 3),
                      padding=(0, 1), stride=(2, 1)),
            nn.LayerNorm([2 * sequence_length, 18, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=2 * sequence_length, out_channels=4 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.LayerNorm([4 * sequence_length, 8, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=4 * sequence_length, out_channels=8 * sequence_length, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.LayerNorm([8 * sequence_length, 3, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=8 * sequence_length, out_channels=16 * sequence_length, kernel_size=3, padding=1),
            nn.LayerNorm([16 * sequence_length, 3, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=16 * sequence_length, out_channels=32 * sequence_length, kernel_size=3),
            nn.LayerNorm([32 * sequence_length, 1, 1]),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2d_o = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=2 * input_channel, kernel_size=(6, 3),
                      padding=(0, 1), stride=(2, 1)),
            nn.LayerNorm([2 * input_channel, 18, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=2 * input_channel, out_channels=4 * input_channel, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.LayerNorm([4 * input_channel, 8, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=4 * input_channel, out_channels=8 * input_channel, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.LayerNorm([8 * input_channel, 3, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=8 * input_channel, out_channels=16 * input_channel, kernel_size=3, padding=1),
            nn.LayerNorm([16 * input_channel, 3, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=16 * input_channel, out_channels=20 * input_channel, kernel_size=3),
            nn.LayerNorm([20 * input_channel, 1, 1]),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2d_y = nn.Sequential(
            nn.Conv2d(in_channels=output_channel, out_channels=8 * output_channel, kernel_size=(6, 3),
                      padding=(0, 1), stride=(2, 1)),
            nn.LayerNorm([8 * output_channel, 18, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=8 * output_channel, out_channels=16 * output_channel, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.LayerNorm([16 * output_channel, 8, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=16 * output_channel, out_channels=40 * output_channel, kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.LayerNorm([40 * output_channel, 3, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=40 * output_channel, out_channels=40 * output_channel, kernel_size=3, padding=1),
            nn.LayerNorm([40 * output_channel, 3, 3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=40 * output_channel, out_channels=160 * output_channel, kernel_size=3),
            nn.LayerNorm([160 * output_channel, 1, 1]),
            nn.LeakyReLU(0.2, True),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * sequence_length, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

    def forward(self,
                seq_x,  # [5, 8, 40, 3]
                # x,  # [8, 40, 3]
                y,  # [2, 40, 3]
                ):
        o = self.conv3d(seq_x)
        o = torch.reshape(o, (o.shape[0], o.shape[2], o.shape[3], o.shape[4]))
        o = self.conv2d_s(o)
        # o_x = self.conv2d_o(x)
        o_y = self.conv2d_y(y)
        s = o + o_y
        s = torch.reshape(s, (s.shape[0], s.shape[1]))

        score = self.fc(s)
        return score


if __name__ == '__main__':
    g = Generator(5, 8, 67, 4)
    d = Discriminator(5, 8, 67, 4)
    i = torch.randn(20, 8, 5, 67, 3)
    a, _ = cnn_paras_count(g)
    # # h = torch.randn(20, 4, 67, 3)
    # e = g(i)
    print((a * 4) / 1024)
    # print(e.shape)
    # y = d(i, e)
    # print(y)
    # # conv_gru = ConvGRULayer(4, 4, 1)
    # # conv_gru(i, h)
