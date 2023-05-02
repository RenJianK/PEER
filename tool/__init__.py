#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : __init__.py.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/7/27 17:37
import time
import numpy as np
import torch


def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params


def judge_road_junction(lane_name: str):
    if lane_name.find('gneE') > -1:
        return True
    return False


def get_lane_number(lane_name: str):
    lane_number = lane_name.split('_')[-1]
    return int(lane_number)


def getFutureVehicle(grid, vid_now, vid_before):
    # [40, 3, 8]
    # search vehicle information of vehicle list
    grid = np.array(grid)
    grid = grid.reshape((-1, grid.shape[-1]))
    x = grid[:, 0]
    y = grid[:, 1]

    # transpose the array
    result = [0 for _ in range(grid.shape[0])]
    result_y = [0 for _ in range(grid.shape[0])]

    n = 0

    for j in range(grid.shape[0]):
        if vid_before[j] != '':
            result[j] = x[vid_now.index(vid_before[j])].item() if vid_before[j] in vid_now else 0
            result_y[j] = (y[vid_now.index(vid_before[j])].item() + 0.5) * 3.2 if vid_before[j] in vid_now else 0
            n += 1 if vid_before[j] in vid_now else 0

    return result, result_y, n


def getFutureVehicleHighD(grid, vid_now, vid_before):
    # search vehicle information of vehicle list
    grid = np.array(grid)
    grid = grid.reshape((-1, grid.shape[-1]))
    x = grid[:, 0]
    y = grid[:, 1]

    # transpose the array
    result = [0 for _ in range(grid.shape[0])]
    result_y = [0 for _ in range(grid.shape[0])]

    n = 0

    for j in range(grid.shape[0]):
        if vid_before[j] != '':
            result[j] = x[vid_now.index(vid_before[j])].item() if vid_before[j] in vid_now else 0
            result_y[j] = y[vid_now.index(vid_before[j])].item() if vid_before[j] in vid_now else 0
            n += 1 if vid_before[j] in vid_now else 0

    return result, result_y, n