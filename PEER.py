#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : PEER.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/9/18 21:49

import json
import math
import os
from random import random, randint
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset.repaintDatasets import RepaintDatasets
from model.PEER_MODEL import *
from model.GroupPredictModel import *

torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.cuda.manual_seed_all(666)


def dataloader(batch_size=500, train=True):
    label = '训练' if train else '测试'
    print("====================================加载{}数据====================================".format(label))
    grip_data = RepaintDatasets(train=train)
    loader = DataLoader(grip_data, batch_size=batch_size)
    print("加载完毕，{}数据数量为{}".format(label, len(grip_data)))
    return loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def cal_grad_penalty(critic, real_samples, fake_samples, x, x_pred=None, gtp=None):
    """计算critic的惩罚项"""

    # 定义alpha
    alpha = torch.Tensor(np.random.randn(real_samples.size(0), 1, 1, 1)).cuda()

    # 从真实数据和生成数据中的连线采样
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).cuda()

    if gtp is None:
        d_interpolates = critic(x, interpolates)
    else:
        d_interpolates = critic(x, interpolates, gtp
                                # , h_global_pred
                                )

    # if x_pred is None:
    #     d_interpolates = critic(x, interpolates, gtp)  # 输出维度：[B, 1]
    # else:
    #     d_interpolates = critic(x, x_pred, interpolates)  # 输出维度：[B, 1]

    fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = fake.cuda()

    d_interpolates = torch.reshape(d_interpolates, (real_samples.size(0), 1))

    # 对采样数据进行求导
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # 返回一个元组(value, )

    gradients = gradients.reshape(gradients.shape[0], -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean().cuda()

    return gradient_penalty


class GANCSR(nn.Module):
    """System model.

    Longer class information....

    Args:
        sequence_length (int): Number of beams to use (see base ``parallel_paths``).
        input_channel (int): See base.
        input_height (int): See base.
        output_channel (int): See base.
        learning_rate (int): See base.
        epochs (int): See base.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, input_channel=10, input_height=40,
                 output_channel=2, learning_rate=0.0004,
                 epochs=10000):
        super().__init__()
        self.config = Config()
        sequence_length = self.config.history_window
        if not os.path.isdir(self.config.path.model_path):
            os.mkdir(self.config.path.model_path)
        self.generator_file = os.path.join(self.config.path.model_path, self.config.generator_path)
        self.discriminator_file = os.path.join(self.config.path.model_path, self.config.discriminator_path)
        self.g_discriminator_file = os.path.join(self.config.path.model_path, self.config.g_discriminator_path)
        self.g_generator_file = os.path.join(self.config.path.model_path, self.config.g_generator_path)

        if os.path.isfile('gtp_model.pth'):
            self.gtp_model = torch.load('gtp_model.pth')

        self.input_height = input_height
        if os.path.isfile(self.generator_file):
            self.generator = torch.load(self.generator_file)
        else:
            self.generator = Generator(
                sequence_length,
                input_channel,
                input_height,
                output_channel
            )

        if os.path.isfile(self.discriminator_file):
            self.discriminator = torch.load(self.discriminator_file)
        else:
            self.discriminator = Discriminator(
                sequence_length, input_channel, input_height, output_channel)

        if os.path.isfile(self.g_generator_file):
            self.global_generator = torch.load(self.g_generator_file)
        else:
            self.global_generator = Global_Generator(sequence_length, input_channel, input_height, output_channel)

        # if os.path.isfile(self.g_discriminator_file):
        #     self.global_discriminator = torch.load(self.g_discriminator_file)
        # else:
        #     self.global_discriminator = Global_Discriminator(sequence_length, input_channel, input_height,
        #                                                      output_channel)

        # self.generator = nn.DataParallel(self.generator)
        # self.discriminator = nn.DataParallel(self.discriminator)

        self.generator.cuda()
        self.discriminator.cuda()
        self.global_generator.cuda()
        # self.global_discriminator.cuda()

        # self.generator.apply(weights_init)
        # self.discriminator.apply(weights_init)
        # self.global_generator.apply(weights_init)
        # self.global_discriminator.apply(weights_init)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            # weight_decay=0.1
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            # weight_decay=1
        )
        # self.opt_global_g = torch.optim.Adam(
        #     self.global_generator.parameters(),
        #     lr=learning_rate
        # )
        # self.opt_global_d = torch.optim.Adam(
        #     self.global_discriminator.parameters(),
        #     lr=learning_rate
        # )

        self.bce_loss = nn.BCELoss().cuda()
        self.l2_loss = nn.MSELoss().cuda()
        self.l1_loss = nn.L1Loss().cuda()

        self.train_data = dataloader(train=True)
        self.test_data = dataloader(train=False)

        self.milestones = [
            100, 200, 300, 700,
            1000, 2000, 4000, 7000]
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.opt_g, milestones=self.milestones, gamma=0.5)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.opt_d, milestones=self.milestones, gamma=0.5)
        # self.scheduler_global_g = torch.optim.lr_scheduler.MultiStepLR(self.opt_global_g, milestones=self.milestones,
        #                                                                gamma=0.5)
        # self.scheduler_global_d = torch.optim.lr_scheduler.MultiStepLR(self.opt_global_d, milestones=self.milestones,
        #                                                                gamma=0.5)

        self.epochs = epochs

    def initial_select_list(self, sequence: np.ndarray):
        select_list = {
            i: 0 for i in range(
                math.ceil(
                    self.input_height /
                    self.config.output_window))
        }

        for i in range(
                math.ceil(
                    self.input_height /
                    self.config.output_window)):
            start = i * self.config.output_window
            end = (i + 1) * self.config.output_window if (i + 1) * \
                                                         self.config.output_window <= self.input_height else self.input_height
            select_list[i] += np.sum(sequence[:, start: end, :, 3])

        return select_list

    def select_area(self, select_list: dict):
        # Select the area currently performing completion according to a certain priority
        m = 0
        index = 0
        for key in select_list:
            if select_list[key] > m and random() < self.config.epsilon and select_list[key] > 0:
                m = select_list[key]
                index = key

        if index - 1 in select_list:
            select_list[index - 1] += 1

        if index + 1 in select_list:
            select_list[index + 1] += 1

        select_list[index] = -100

        return index, select_list

    @staticmethod
    def need_del(sequence: np.ndarray):
        """
        Delete the sequence without unperceived areas to best train our model.

        Returns:
            need_del_list: list
            List of sequences need delete
        """
        need_del_list = []
        for i in range(sequence.shape[0]):
            if np.sum(sequence[i, :, :, 4]) == 0:
                need_del_list.append(i)

        return need_del_list

    @staticmethod
    def del_tensor(need_del_list: list, sequence: torch.Tensor):
        if len(need_del_list) == 0:
            return sequence
        res = np.delete(sequence.detach().cpu().numpy(), np.array(need_del_list), axis=0)

        return torch.tensor(res)

    def predict_segment(self, x: torch.Tensor, y: torch.Tensor, select_list: dict, real_scenario, fake_scenario,
                        gtp,
                        # h_global_pred,
                        train=False):
        """
        Args:
            x: Tensor
            The road information, shape: [number, seq_len, length, width, 8]
            y: Tensor
            The global view of road, shape: [number, length, width, output_dim]
            select_list: dict
            Sequences of the order in road grid map
            real_scenario:
            fake_scenario:
            train: boolean

        Returns:
            x_i: Tensor
            Input of generator, shape: [number, seq_len, length, width, 8]
            y_i: Tensor
            Real road information in the unperceived areas, shape: [number, window, width, output_dim]
            x: Tensor
            Road information merged the predicted result for next iteration, shape: [number, seq_len, length, width, 8]
            y_pred: Tensor
            Predicted result, shape: [number, window, width, output_dim]
            need_del_list: list
            The sequence without unperceived areas to best train our model.
            select_list: dict
            The updated select list.
        """
        index, select_list = self.select_area(select_list)
        x_i = x.detach()
        start = index * self.config.output_window
        end = (index + 1) * self.config.output_window if (index + 1) * \
                                                         self.config.output_window <= self.input_height \
            else self.input_height
        x_i[:, :, start: end, :, -3] = 1
        x_i = x_i.cuda()
        x_i = torch.permute(x_i, (0, 4, 1, 2, 3))

        y_i = torch.zeros(
            (x.shape[0], self.config.output_window, 3, y.shape[-1])).cuda()
        y_i[:, 0: end - start] = y.detach()[:, start: end]

        temporal_x_i = x.detach()
        # mask the area where is visible
        temporal_x_i = torch.permute(
            temporal_x_i, (1, 0, 2, 3, 4))[-1]  # [batch, 40, 3, 8]
        y_i[temporal_x_i[:, start: end, :, 4] == 0] = 0
        seg_x = x_i.clone()[:, :, :, start: end, :]
        seg_x = torch.permute(seg_x, (2, 0, 1, 3, 4))[-1]  # [batch, 8, len, 3]
        seg_x = seg_x.cuda()

        _, y_pred = self.generator(x_i, seg_x
                                   , gtp
                                   # , h_global_pred
                                   )  # [32, 2, 10, 3]

        # add the predicted result where is invisible into the next input
        temporal_y = y_pred.detach().cpu() if not train else y.detach().numpy()[:, start:end]
        if not train:
            temporal_y = torch.permute(temporal_y, (0, 2, 3, 1)).numpy()

        # temporal_y = y.detach().numpy()[:, start: end]
        temporal_x = temporal_x_i.numpy()
        temporal_x[:, start: end][temporal_x[:, start: end, :, 4] ==
                                  1, 2] = temporal_y[temporal_x[:, start: end, :, 4] == 1, 0]
        temporal_x[:, start: end][temporal_x[:, start: end, :, 4] ==
                                  1, 3] = temporal_y[temporal_x[:, start: end, :, 4] == 1, 1] / 100
        x = torch.permute(x, (1, 0, 2, 3, 4))
        x[-1] = torch.tensor(temporal_x)
        x = torch.permute(x, (1, 0, 2, 3, 4))
        y_i = torch.permute(y_i, (0, 3, 1, 2))
        index = np.array([i for i in range(real_scenario.shape[0])])
        real_scenario[index, :, start:end] = y_i.detach().cpu().numpy()
        fake_scenario[index, :, start:end] = y_pred.detach().cpu().numpy()

        return x_i, y_i, x, y_pred, select_list

    def combine_result(
            self, original_scene: torch.Tensor, local_repainted: np.ndarray, global_repainted:
            np.ndarray
    ) -> torch.Tensor:
        """
        merge the local repainted result and global repainted result
        rule of merge: make the local repainted result if there is no
        vehicle in this RUPA else global repainted result

        Args:
            original_scene: shape [batch, height, width, channel]
                initial road information
            local_repainted: shape [batch, 2, height, width]
                repainted result by local generator
            global_repainted: shape [batch, 2, height, width]
                repainted result by global generator

        Returns:
            combined_result: shape [batch, 2, height, width]
                finally repainted result by combine the local and global result

        """
        combined_result = torch.zeros(local_repainted.shape)

        for i in range(original_scene.shape[0]):
            # traverse each RUPA to check whether is there a vehicle
            plain_info = original_scene[i].numpy()
            for j in range(self.input_height // self.config.output_window):
                start = j * self.config.output_window
                end = (j + 1) * self.config.output_window
                combined_result[i, :, start: end] = torch.Tensor(local_repainted[i, :, start:end]) if np.sum(
                    plain_info[start:end, :, 3]) != 0 else torch.Tensor(global_repainted[i, :, start: end])

        return combined_result

    def weighted_hit_rate_v1(
            self, original_scene: torch.Tensor, repainted: np.ndarray, true_scenario: np.ndarray
    ) -> float:
        """
        compute weighted hit rate, proportion: 1: 1

        Args:
            original_scene: [batch, height, width]
            repainted: [batch, height, width]
            true_scenario: [batch, height, width]

        Returns:

        """
        hit_rate = 0
        for j in range(self.input_height // self.config.output_window):
            start = j * self.config.output_window
            end = (j + 1) * self.config.output_window
            a = np.sum(repainted[:, start:end])
            b = np.sum(true_scenario[:, start:end])
            c = torch.sum(original_scene[:, start: end])

            hit_rate += (a + c) / (b + c) if ((a + c) / (b + c)) < 1 else (b + c) / (a + c)

        return hit_rate / (self.input_height // self.config.output_window)

    def weighted_hit_of_line(
            self, original_scene: torch.Tensor, repainted: np.ndarray, true_scenario: np.ndarray
    ) -> float:
        hit_rate = 0
        for j in range(self.input_height // self.config.output_window):
            start = j * self.config.output_window
            end = (j + 1) * self.config.output_window

            hit_line_rate = 0
            for k in range(self.config.max_lane_num):
                a = np.sum(repainted[:, start:end, k])
                b = np.sum(true_scenario[:, start:end, k])
                c = torch.sum(original_scene[: start: end, k])

                hit_line_rate += (a + c) / (b + c) if ((a + c) / (b + c)) < 1 else (b + c) / (a + c)

            hit_rate += hit_line_rate / self.config.max_lane_num

        return hit_rate / (self.input_height // self.config.output_window)

    def generate_grid(self, x, y):

        grid = torch.zeros((x.shape[0], 40, 3, 3)).cuda()

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] != 0:
                    v_x, v_y = x[i, j], y[i, j]
                    x_index, y_index = int(v_x.item() // (self.config.min_gap + self.config.min_length)), \
                                       int(v_y.item() // 3.2)
                    x_index = x_index if x_index < self.input_height else self.input_height - 1
                    y_index = y_index if y_index < 3 else 2
                    # print(x_index, y_index)
                    grid[i, x_index, y_index, 0] = v_x
                    grid[i, x_index, y_index, 1] = y_index
                    grid[i, x_index, y_index, 2] = 100

        grid = torch.permute(grid, (0, 3, 1, 2))
        return grid

    def train_model(self):
        print(
            "====================================train model====================================")
        # self.global_generator.eval()
        self.gtp_model.eval()
        e = 0
        for epoch in range(self.epochs):
            # self.gtp_model.eval()

            t1 = time.time()
            mse = 0
            g_mse = 0
            combined_mse = 0
            d_loss = 0
            d_global_loss = 0
            g_error = 0
            s_loss = 0
            hit_rate = 0
            p_loss = 0

            n = 0
            # e = e + 1 if e <= 3 else 1
            # f = [i for i in range(len(self.train_data))]
            self.generator.train()
            self.discriminator.train()

            for index, data in enumerate(self.train_data):
                # if index not in f[e - 1::3]:
                #     continue
                x, y = data  # [32, 5, 40, 3, 8], [32, 40, 3, output_dim]
                n += 1
                c_x = torch.permute(x, (1, 0, 2, 3, 4))[-2]
                c_x = c_x[:, :, :, 3]  # [32, 40, 3]
                c_x = c_x.reshape(c_x.shape[0], -1)
                original_x = torch.permute(x, (1, 0, 2, 3, 4))[-1]  # [32, 40, 3, 8]
                group_info = torch.zeros((x.shape[0], x.shape[1] - 1, x.shape[2], x.shape[3], x.shape[4] - 2))
                group_info[:, :, :, :, :3], group_info[:, :, :, :, 3:] = x[:, :-1, :, :, :3], x[:, :-1, :, :, 5:]
                group_info = group_info.cuda()
                group_info = torch.permute(group_info, (0, 4, 1, 2, 3))

                # The prediction is centered on the place where v2x vehicles exist
                sequence = torch.permute(x, (1, 0, 2, 3, 4))[-1]
                select_list = self.initial_select_list(sequence.detach().numpy())
                real_scenario = np.zeros((x.shape[0], 2, self.input_height, 3))
                fake_scenario = np.zeros((x.shape[0], 2, self.input_height, 3))

                res_x = x.detach()
                res_x = res_x.cuda()
                y_global = y.detach()
                y_global = y_global.cuda()

                y_global[original_x[:, :, :, 4] == 0] = 0
                res_x = torch.permute(res_x, (0, 4, 1, 2, 3))
                y_global = torch.permute(y_global, (0, 3, 1, 2))
                with torch.no_grad():
                    h_global_pred, y_global_pred = self.global_generator(res_x)

                # d_global_real = torch.mean(self.global_discriminator(res_x, y_global))
                # d_global_fake = torch.mean(self.global_discriminator(res_x, y_global_pred.detach()))
                #
                # gradient_penalty = cal_grad_penalty(self.global_discriminator, y_global, y_global_pred.detach(),
                #                                     res_x)
                # l_d = d_global_fake - d_global_real + gradient_penalty * 1.5
                #
                # self.opt_global_d.zero_grad()
                # l_d.backward()
                # self.opt_global_d.step()

                # d_global_fake_2 = self.global_discriminator(res_x, y_global)
                # p_pred = y_global_pred[:, 1]
                # s_pred = y_global_pred[:, 0]
                # p_real = y_global[:, 1]
                # s_real = y_global[:, 0]
                #
                # p_pred = torch.reshape(p_pred, (p_pred.shape[0], -1))
                # s_pred = torch.reshape(s_pred, (s_pred.shape[0], -1))
                # p_real = torch.reshape(p_real, (p_real.shape[0], -1))
                # s_real = torch.reshape(s_real, (s_real.shape[0], -1))
                #
                # # g_mse_p_loss = self.l2_loss(p_pred, p_real)
                # # g_mse_s_loss = self.l2_loss(s_pred, s_real)
                # # g_ad_loss = -torch.mean(d_global_fake_2)
                # # g_loss = g_mse_s_loss + g_mse_p_loss + g_ad_loss
                #
                # self.opt_global_g.zero_grad()
                # g_loss.backward()
                # self.opt_global_g.step()
                # g_mse += g_mse_p_loss

                # d_global_loss += l_d
                temporal_y = torch.permute(y_global_pred, (0, 2, 3, 1)).detach().cpu().numpy()
                temporal_x = original_x.detach().numpy()

                temporal_x[:, :][temporal_x[:, :, :, 4] ==
                                 1, 2] = temporal_y[temporal_x[:, :, :, 4] == 1, 0]
                temporal_x[:, :][temporal_x[:, :, :, 4] ==
                                 1, 3] = temporal_y[temporal_x[:, :, :, 4] == 1, 1]
                x = torch.permute(x, (1, 0, 2, 3, 4))
                x[-1] = torch.tensor(temporal_x)
                x = torch.permute(x, (1, 0, 2, 3, 4))
                gtp_x, gtp_y = self.gtp_model(group_info)

                gtp_x[c_x == 0] = 0
                gtp_y[c_x == 0] = 0

                gtp_grid = self.generate_grid(gtp_x.detach(), gtp_y.detach())

                for i in range(
                        math.ceil(
                            self.input_height /
                            self.config.output_window)
                ):
                    # print(x.shape)
                    x_i, y_i, x, y_pred, select_list = self.predict_segment(x, y, select_list,
                                                                            real_scenario,
                                                                            fake_scenario,
                                                                            gtp_grid,
                                                                            # h_global_pred.detach(),
                                                                            train=True)

                    d_real = torch.mean(self.discriminator(x_i, y_i
                                                           , gtp_grid
                                                           # , h_global_pred.detach())
                                                           ))
                    d_fake = torch.mean(self.discriminator(x_i, y_pred.detach()
                                                           , gtp_grid,
                                                           # h_global_pred.detach())
                                                           ))

                    gradient_penalty = cal_grad_penalty(self.discriminator, y_i, y_pred.detach(), x_i,
                                                        # h_global_pred.detach(),
                                                        gtp=gtp_grid
                                                        )
                    l_d = d_fake - d_real + gradient_penalty * 1.5

                    self.opt_d.zero_grad()
                    l_d.backward()
                    self.opt_d.step()

                    d_loss += l_d
                    d_fake_2 = self.discriminator(x_i, y_pred
                                                  , gtp_grid
                                                  # , h_global_pred.detach()
                                                  )

                    # print(y_pred.shape)
                    p_pred = y_pred[:, 1]
                    s_pred = y_pred[:, 0]
                    p_real = y_i[:, 1]
                    s_real = y_i[:, 0]

                    p_pred = torch.reshape(p_pred, (p_pred.shape[0], -1))
                    s_pred = torch.reshape(s_pred, (s_pred.shape[0], -1))
                    p_real = torch.reshape(p_real, (p_real.shape[0], -1))
                    s_real = torch.reshape(s_real, (s_real.shape[0], -1))
                    # real = torch.reshape(y_i, (y_i.shape[0], -1))
                    # fake = torch.reshape(y_pred, (y_pred.shape[0], -1))

                    g_mse_p_loss = self.l2_loss(p_pred, p_real)
                    g_mse_s_loss = self.l2_loss(s_pred, s_real)
                    # g_mse = self.l2_loss(real, fake)
                    g_ad_loss = -torch.mean(d_fake_2)
                    g_loss = g_mse_s_loss + g_mse_p_loss + g_ad_loss

                    self.opt_g.zero_grad()
                    g_loss.backward()
                    self.opt_g.step()

                    g_error += g_loss

                    mse += g_mse_p_loss

                # combined_result = self.combine_result(original_x, y_global_pred.detach().cpu().numpy(), fake_scenario)
                # fake_scenario = combined_result.numpy()
                # p_pred = combined_result[:, 1]
                # p_real = torch.Tensor(real_scenario)[:, 1]

                p_pred = torch.reshape(p_pred, (p_pred.shape[0], -1))
                p_real = torch.reshape(p_real, (p_real.shape[0], -1))
                combined_mse += self.l2_loss(p_pred, p_real)

                fake_scenario[:, 1][fake_scenario[:, 1] >= 50] = 100
                fake_scenario[:, 1][fake_scenario[:, 1] < 50] = 0
                hit_rate += self.weighted_hit_of_line(original_x[:, :, :, 3], fake_scenario[:, 1], real_scenario[:, -1])

                fake_position_scenario = np.reshape(fake_scenario[:, 1], (fake_scenario[:, 1].shape[0], -1))
                real_position_scenario = np.reshape(real_scenario[:, 1], (real_scenario[:, 1].shape[0], -1))
                # p = np.reshape(original_x[:, :, :, 3].detach().numpy(), (original_x.shape[0], -1))
                #
                # cross_section = np.sqrt(np.multiply(fake_position_scenario, real_position_scenario))
                # c = np.sum(p, axis=1)
                # a = np.sum(cross_section, axis=1)
                # b = np.sum(real_position_scenario, axis=1)
                # n_d = np.where(b == 0)
                # a = np.delete(a, n_d)
                # b = np.delete(b, n_d)
                # c = np.delete(c, n_d)

                # hit_rate += np.mean((a + c) / (b + c))

                p_loss += np.mean(
                    np.mean(np.abs(fake_position_scenario - real_position_scenario))
                )

            # with open('train.json', 'w') as f:
            #     json.dump(
            #         {
            #             'position':
            #                 {
            #                     'real': y.detach().cpu().numpy()[2, 1].tolist(),
            #                     'fake': y_pred.detach().cpu().numpy()[2, 1].tolist()
            #                 },
            #             'speed':
            #                 {
            #                     'real': y.detach().cpu().numpy()[2, 0].tolist(),
            #                     'fake': y_pred.detach().cpu().numpy()[2, 0].tolist()
            #                 },
            #         }, f)

            print(
                "epoch is {} | {}, the mse loss is {:.3f}, the d loss is {:.3f}, the g loss is , the speed loss"
                " is {:.3f}, the position loss is {:.3f}, the hit rate is {:.6f}, the g_mse is {:.3f}, the "
                "d_global_loss is {:.3f}, combined mse is {:.3f}".
                format(epoch, self.epochs, mse / (n * math.ceil(self.input_height / self.config.output_window)),
                       d_loss / (n * math.ceil(self.input_height / self.config.output_window)),
                       # g_loss / (n * math.ceil(self.input_height / self.config.output_window)),
                       s_loss / n, p_loss / n, hit_rate / n, g_mse / n, d_global_loss / n, combined_mse / n)
                , end='\n'
            )

            self.scheduler_g.step()
            self.scheduler_d.step()
            # self.scheduler_global_d.step()
            # self.scheduler_global_g.step()

            if epoch % 50 == 0 and epoch > 0:
                torch.save(self.generator, self.generator_file)
                torch.save(self.discriminator, self.discriminator_file)
                # torch.save(self.global_generator, self.g_generator_file)
                # torch.save(self.global_discriminator, self.g_discriminator_file)

            # if epoch % 5 == 0:
            self.test()

            t2 = time.time()
            print(t2 - t1)

    def test(self):
        p_loss = 0
        s_loss = 0
        hit_rate = 0
        g_mse = 0
        d_global_loss = 0
        combined_mse = 0
        n = 0
        self.global_generator.eval()
        self.generator.train()
        self.gtp_model.eval()

        for data in self.test_data:
            with torch.no_grad():
                x, y = data  # [32, 5, 40, 3, 8], [32, 40, 3, output_dim]
                n += 1
                sequence = torch.permute(x, (1, 0, 2, 3, 4))[-1]
                select_list = self.initial_select_list(sequence.detach().numpy())
                real_scenario = np.zeros((x.shape[0], 2, self.input_height, 3))
                fake_scenario = np.zeros((x.shape[0], 2, self.input_height, 3))
                c_x = torch.permute(x, (1, 0, 2, 3, 4))[-2]
                c_x = c_x[:, :, :, 3]  # [32, 40, 3]
                c_x = c_x.reshape(c_x.shape[0], -1)
                original_x = torch.permute(x, (1, 0, 2, 3, 4))[-1]  # [32, 40, 3, 8]
                group_info = torch.zeros((x.shape[0], x.shape[1] - 1, x.shape[2], x.shape[3], x.shape[4] - 2))
                group_info[:, :, :, :, :3], group_info[:, :, :, :, 3:] = x[:, :-1, :, :, :3], x[:, :-1, :, :, 5:]
                group_info = group_info.cuda()
                group_info = torch.permute(group_info, (0, 4, 1, 2, 3))
                res_x = x.detach()
                res_x = res_x.cuda()
                y_global = y.detach()
                y_global = y_global.cuda()

                y_global[original_x[:, :, :, 4] == 0] = 0
                res_x = torch.permute(res_x, (0, 4, 1, 2, 3))
                y_global = torch.permute(y_global, (0, 3, 1, 2))

                h_global_pred, y_global_pred = self.global_generator(res_x)

                p_pred = y_global_pred[:, 1]
                p_real = y_global[:, 1]

                p_pred = torch.reshape(p_pred, (p_pred.shape[0], -1))
                p_real = torch.reshape(p_real, (p_real.shape[0], -1))

                g_mse_p_loss = self.l2_loss(p_pred, p_real)
                g_mse += g_mse_p_loss

                temporal_y = torch.permute(y_global_pred, (0, 2, 3, 1)).detach().cpu().numpy()
                temporal_x = original_x.numpy()
                temporal_x[:, :][temporal_x[:, :, :, 4] ==
                                 1, 2] = temporal_y[temporal_x[:, :, :, 4] == 1, 0]
                temporal_x[:, :][temporal_x[:, :, :, 4] ==
                                 1, 3] = temporal_y[temporal_x[:, :, :, 4] == 1, 1]
                x = torch.permute(x, (1, 0, 2, 3, 4))
                x[-1] = torch.tensor(temporal_x)
                x = torch.permute(x, (1, 0, 2, 3, 4))

                gtp_x, gtp_y = self.gtp_model(group_info)

                gtp_x[c_x == 0] = 0
                gtp_y[c_x == 0] = 0

                gtp_grid = self.generate_grid(gtp_x.detach(), gtp_y.detach())
                # local_h = torch.zeros(h_global_pred.shape).cuda()
                for i in range(
                        math.ceil(
                            self.input_height /
                            self.config.output_window)
                ):
                    x_i, y_i, x, y_pred, select_list = self.predict_segment(x, y, select_list,
                                                                            real_scenario,
                                                                            fake_scenario,
                                                                            gtp_grid,
                                                                            train=False)

                    p_pred = y_pred[:, 1]
                    s_pred = y_pred[:, 0]
                    p_real = y_i[:, 1]
                    s_real = y_i[:, 0]

                    p_pred = torch.reshape(p_pred, (p_pred.shape[0], -1))
                    s_pred = torch.reshape(s_pred, (s_pred.shape[0], -1))
                    p_real = torch.reshape(p_real, (p_real.shape[0], -1))
                    s_real = torch.reshape(s_real, (s_real.shape[0], -1))

                    g_mse_p_loss = self.l2_loss(p_pred, p_real)
                    g_mse_s_loss = self.l2_loss(s_pred, s_real)

                    s_loss += g_mse_s_loss
                    p_loss += g_mse_p_loss

                # res_x = x.detach()
                # res_x[:, -1] = original_x.detach()
                # x_pred = torch.permute(x, (1, 0, 2, 3, 4))[-1]

                # res_x = res_x.cuda()
                # x_pred = x_pred.cuda()
                # y = y.cuda()

                # y[original_x[:, :, :, 4] == 0] = 0
                # res_x = torch.permute(res_x, (0, 4, 1, 2, 3))
                # x_pred = torch.permute(x_pred, (0, 3, 1, 2))
                # y = torch.permute(y, (0, 3, 1, 2))

                # y_global_pred = self.global_generator(res_x)

                # p_pred = y_global_pred[:, 1]
                # p_real = y[:, 1]
                # combined_result = self.combine_result(original_x, y_global_pred.detach().cpu().numpy(), fake_scenario)
                # fake_scenario = combined_result.numpy()

                # p_pred = fake_scenario[:, 1]
                # p_real = torch.Tensor(real_scenario)[:, 1]

                p_pred = torch.reshape(p_pred, (p_pred.shape[0], -1))
                p_real = torch.reshape(p_real, (p_real.shape[0], -1))
                combined_mse += self.l2_loss(p_pred, p_real)

                # g_mse_p_loss = self.l2_loss(p_pred, p_real)

                # g_mse += g_mse_p_loss.item()

                # fake_scenario = y_global_pred.detach().cpu().numpy()
                # real_scenario = y.detach().cpu().numpy()

                fake_scenario[:, 1][fake_scenario[:, 1] > 50] = 100
                fake_scenario[:, 1][fake_scenario[:, 1] <= 50] = 0

                fake_position_scenario = np.reshape(fake_scenario[:, 1], (fake_scenario[:, 1].shape[0], -1))
                real_position_scenario = np.reshape(real_scenario[:, 1], (real_scenario[:, 1].shape[0], -1))
                p = np.reshape(original_x[:, :, :, 3].detach().numpy(), (original_x.shape[0], -1))

                cross_section = np.sqrt(np.multiply(fake_position_scenario, real_position_scenario))
                c = np.sum(p, axis=1)
                a = np.sum(cross_section, axis=1)
                b = np.sum(real_position_scenario, axis=1)
                n_d = np.where(b == 0)
                a = np.delete(a, n_d)
                b = np.delete(b, n_d)
                c = np.delete(c, n_d)

                hit_rate += np.mean((a + c) / (b + c))

                # hit_rate += self.weighted_hit_rate_v1(original_x[:, :, :, 3], fake_scenario[:, 1], real_scenario[:, -1])
                # hit_rate += self.weighted_hit_of_line(original_x[:, :, :, 3], fake_scenario[:, 1], real_scenario[:, -1])

        print(
            "test position loss is {:.3f}, the speed loss is {:.3f}, global position loss is {:.3f},the hit rate is {:.3f}".
            format(
                p_loss / (n * math.ceil(self.input_height / self.config.output_window)),
                s_loss / (n * math.ceil(self.input_height / self.config.output_window)),
                g_mse / n,
                hit_rate / n,
            )
        )


if __name__ == '__main__':
    gan_csr = GANCSR()
    gan_csr.train_model()
    # gan_csr.test()
