#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : TinyGAN.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/7/27 17:36

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

LayerNum = 6
BasicMultiple = 2


class Tiny_Conv3D_G(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # 3d conv + residual
        self.s = nn.Sequential(
            nn.Conv3d(in_channels=input_channel, out_channels=input_channel * BasicMultiple, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.BatchNorm3d(input_channel * BasicMultiple),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * BasicMultiple, out_channels=input_channel * 2 * BasicMultiple,
                      padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.BatchNorm3d(input_channel * 2 * BasicMultiple),
            nn.ReLU(True),
        )

        self.c = nn.ModuleList(
            [nn.Sequential(
                nn.Conv3d(in_channels=input_channel * 2 * BasicMultiple, out_channels=input_channel * 2 * BasicMultiple,
                          padding=(1, 1, 1),
                          kernel_size=(3, 3, 3),
                          stride=1)
                , nn.BatchNorm3d(input_channel * 2 * BasicMultiple)
            ) for _ in range(LayerNum)])

        self.e = nn.Sequential(
            nn.Conv3d(in_channels=input_channel * 2 * BasicMultiple, out_channels=input_channel * 1 * BasicMultiple,
                      padding=(1, 1, 1),
                      kernel_size=(3, 3, 3),
                      stride=1),
            nn.BatchNorm3d(input_channel * 1 * BasicMultiple),
            nn.ReLU(True),
            nn.Conv3d(in_channels=input_channel * 1 * BasicMultiple, out_channels=input_channel, padding=(1, 1, 1),
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


class Tiny_Conv3D_D(nn.Module):
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
                           ) for _ in range(LayerNum)])

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


class Tiny_ConvGRUCell(nn.Module):
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


class Tiny_ConvGRULayer(nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel):
        super().__init__()
        self.hidden_size = hidden_channel
        self.out_channel = output_channel
        self.gru_cell = Tiny_ConvGRUCell(input_channel, hidden_channel, output_channel)
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


class Tiny_Conv2D(nn.Module):
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


class Tiny_Generator(nn.Module):
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
        self.conv3d = Tiny_Conv3D_G(input_channel, input_channel)
        # [batch, t, 4, 67, 3]
        self.conv_gru = Tiny_ConvGRULayer(input_channel, input_channel, 1)
        self.conv2d_s = nn.Sequential(
            nn.Conv2d(in_channels=sequence_length, out_channels=BasicMultiple * sequence_length, kernel_size=(6, 3),
                      padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(BasicMultiple * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=BasicMultiple * sequence_length, out_channels=BasicMultiple * 2 * sequence_length,
                      kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(BasicMultiple * 2 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=BasicMultiple * 2 * sequence_length, out_channels=BasicMultiple * 4 * sequence_length,
                      kernel_size=(4, 3),
                      padding=(0, 1)
                      , stride=(2, 1)),
            nn.BatchNorm2d(BasicMultiple * 4 * sequence_length),
            nn.ReLU(True),
        )
        self.conv2d_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=BasicMultiple * 4 * sequence_length,
                          out_channels=BasicMultiple * 4 * sequence_length, kernel_size=3, padding=1),
                nn.BatchNorm2d(BasicMultiple * 4 * sequence_length),
            )
            for _ in range(LayerNum)
        ])

        self.conv2d_e = nn.Sequential(
            nn.Conv2d(in_channels=BasicMultiple * 4 * sequence_length, out_channels=BasicMultiple * 8 * sequence_length,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(BasicMultiple * 8 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=BasicMultiple * 8 * sequence_length, out_channels=32 * sequence_length,
                      kernel_size=3),
            nn.BatchNorm2d(32 * sequence_length),
            nn.ReLU(True)
        )

        self.transCNN = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * sequence_length * 3, out_channels=output_channel * 8, kernel_size=4,
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
            nn.Sigmoid()
        )

        self.seg = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=BasicMultiple * 2 * sequence_length, kernel_size=3,
                      padding=(0, 1)),
            nn.BatchNorm2d(BasicMultiple * 2 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=BasicMultiple * 2 * sequence_length, out_channels=BasicMultiple * 4 * sequence_length,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(BasicMultiple * 4 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=BasicMultiple * 4 * sequence_length, out_channels=BasicMultiple * 8 * sequence_length,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(BasicMultiple * 8 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=BasicMultiple * 8 * sequence_length, out_channels=32 * sequence_length,
                      kernel_size=3),
            nn.BatchNorm2d(32 * sequence_length),
            nn.ReLU(True),
        )

        self.group_fc = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 32 * sequence_length),
            nn.ReLU()
        )
        # self.avg_3d = nn.AdaptiveAvgPool3d(1)
        # self.avg_2d = nn.AdaptiveAvgPool2d(1)
        # self.attention_fc_3d = nn.Sequential(
        #     nn.Linear(10, 20, bias=False),
        #     nn.ReLU(True),
        #     nn.Linear(20, 10, bias=False),
        #     nn.Sigmoid()
        # )
        # self.attention_fc_2d = nn.Sequential(
        #     nn.Linear(5, 10, bias=False),
        #     nn.ReLU(True),
        #     nn.Linear(10, 5, bias=False),
        #     nn.Sigmoid()
        # )
        # self.shallow_3d = nn.Sequential(
        #     nn.Conv3d(in_channels=input_channel, out_channels=input_channel,
        #                             padding=(1, 1, 1),
        #                             kernel_size=(3, 3, 3),
        #                             stride=1),
        #     nn.ReLU(True),
        #     nn.Conv3d(in_channels=input_channel, out_channels=1,
        #               padding=(1, 1, 1),
        #               kernel_size=(3, 3, 3),
        #               stride=1),
        #     nn.ReLU(True)
        # )
        # self.conv = Tiny_Conv2D(input_channel, 1)

    def forward(
            self,
            x,
            seg_x,
            group_predicted_x,
            is_train='train'
    ):
        if is_train == 'test':
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
            o_group_x = self.group_fc(group_predicted_x)
            o_group_x = torch.reshape(o_group_x, (o_group_x.shape[0], o_group_x.shape[1], 1, 1))
            o = torch.cat((o, o_group_x, o_seg_x), dim=1)
            y = self.transCNN(o)
            return y

        if is_train == 'train':
            o = self.conv3d(x)
            o_seg_x = self.seg(seg_x)
            o_group_x = self.group_fc(group_predicted_x)
            o_group_x = torch.reshape(o_group_x, (o_group_x.shape[0], o_group_x.shape[1], 1, 1))

            # compute the first shallow
            # attention
            o_a = self.avg_3d(o).view(o.shape[0], o.shape[1])
            o_a = self.attention_fc_3d(o_a).view(o_a.shape[0], o_a.shape[1], 1, 1, 1)
            o_a = o * o_a.expand_as(o)
            # first shallow
            o_a = self.shallow_3d(o_a)
            o_a = torch.permute(o_a, (0, 2, 1, 3, 4))
            o_a = torch.reshape(o_a, (o_a.shape[0], o_a.shape[1], o_a.shape[3], o_a.shape[4]))  # [batch, 5, 40, 3]
            o_a = relu(self.conv2d_s(o_a))
            for i in range(len(self.conv2d_c) // 2):
                y = self.conv2d_c[i](o_a)
                o_a = relu(o_a + y)
            o_a = relu(self.conv2d_e(o_a))
            o_a = torch.cat((o_a, o_group_x, o_seg_x), dim=1)
            hidden_shallow_1 = o_a
            shallow_1 = self.transCNN(o_a)

            o = torch.permute(o, (0, 2, 1, 3, 4))
            o, h = self.conv_gru(o)  # [batch, 5, 1, 40, 3]
            o = torch.reshape(o, (o.shape[0], o.shape[1], o.shape[3], o.shape[4]))  # [batch, 5, 40, 3]
            # compute the second shallow
            # attention
            o_a_2 = self.avg_2d(o).view(o.shape[0], o.shape[1])
            o_a_2 = self.attention_fc_2d(o_a_2).view(o_a_2.shape[0], o_a_2.shape[1], 1, 1)
            o_a_2 = o * o_a_2.expand_as(o)  # [batch, 5, 40, 3]
            # second shallow
            o_a_2 = relu(self.conv2d_s(o_a_2))
            for i in range(len(self.conv2d_c) // 2):
                y = self.conv2d_c[i](o_a_2)
                o_a_2 = relu(o_a_2 + y)
            o_a_2 = relu(self.conv2d_e(o_a_2))
            o_a_2 = torch.cat((o_a_2, o_group_x, o_seg_x), dim=1)
            hidden_shallow_2 = o_a_2
            shallow_2 = self.transCNN(o_a_2)

            o = relu(self.conv2d_s(o))
            for i in range(len(self.conv2d_c)):
                y = self.conv2d_c[i](o)
                o = relu(o + y)
            o = relu(self.conv2d_e(o))

            o = torch.cat((o, o_group_x, o_seg_x), dim=1)
            y = self.transCNN(o)

            return o, y, shallow_1, shallow_2, hidden_shallow_1, hidden_shallow_2


class Tiny_Global_Generator(nn.Module):
    def __init__(self, sequence_length, input_channel, input_height, output_channel):
        super().__init__()
        self.conv3d = Tiny_Conv3D_G(input_channel, 1)
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
            for _ in range(LayerNum)
        ])
        self.conv2d_s_e = nn.Sequential(
            nn.Conv2d(in_channels=8 * sequence_length, out_channels=16 * sequence_length, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * sequence_length),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16 * sequence_length, out_channels=32 * sequence_length, kernel_size=3),
            nn.BatchNorm2d(32 * sequence_length),
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
            for _ in range(LayerNum)
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
        print(o.shape)
        o = self.conv2d_s_s(o)  # [160, 1, 1]

        for i in range(len(self.conv2d_s_c)):
            y = self.conv2d_s_c[i](o)
            o = relu(o + y)

        o = self.conv2d_s_e(o)

        # o_o = self.conv2d_o_s(x)  # [160, 1, 1]
        #
        # for i in range(len(self.conv2d_o_c)):
        #     y = self.conv2d_o_c[i](o_o)
        #     o_o = relu(o_o + y)
        #
        # o_o = self.conv2d_o_e(o_o)

        y = self.transCNN(o)  # [2, 40, 3]

        return o, y


class Tiny_Discriminator(nn.Module):
    def __init__(self, sequence_length, input_channel, input_height, output_channel):
        super().__init__()
        self.conv3d = Tiny_Conv3D_D(input_channel, input_channel)
        self.conv_gru = Tiny_ConvGRULayer(input_channel, input_channel, 1)
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
            nn.Linear(input_height + 64 * 2 - 2, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

        self.group_fc = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Linear(300, 64 * output_channel),
        )

    def forward(self, x, y, group_predicted_x):
        o = self.conv3d(x)
        o = torch.permute(o, (0, 2, 1, 3, 4))
        o_y = self.res(y)

        o, h = self.conv_gru(o)
        o = torch.reshape(o, (o.shape[0], o.shape[1], o.shape[3], o.shape[4]))
        o = self.conv2d(o)
        o = torch.reshape(o, (o.shape[0], o.shape[2]))

        o_group_x = self.group_fc(group_predicted_x)

        o = torch.cat((o, o_group_x, o_y), dim=-1)

        y = self.fc(o)
        return y


class Tiny_Global_Discriminator(nn.Module):
    def __init__(self, sequence_length, input_channel, input_height, output_channel):
        super().__init__()
        self.conv3d = Tiny_Conv3D_D(input_channel, 1)
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
            nn.Conv2d(in_channels=40 * output_channel, out_channels=80 * output_channel, kernel_size=3),
            nn.LayerNorm([80 * output_channel, 1, 1]),
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
    g = Tiny_Generator(5, 10, 40, 1)
    i = torch.randn(20, 10, 5, 40, 3)
    a, _ = cnn_paras_count(g)
    # # h = torch.randn(20, 4, 67, 3)
    # e = g(i)
    print((a * 4) / 1024)
    # print(e.shape)
    # y = d(i, e)
    # print(y)
    # # conv_gru = ConvGRULayer(4, 4, 1)
    # # conv_gru(i, h)
