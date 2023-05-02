#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : realTrajectory.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/9/3 21:06
import json
import os

from config import Config
from data_process.prepare_data import PrepareData


class RealTrajectory(object):
    """Summary of class here.

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

    def __init__(self):
        self.config = Config()  # basic configuration
        # self.original_data = PrepareData(load=True)
        #
        # self.vehicle_file = os.path.join(
        #     self.config.path.data_path,
        #     self.config.vehicle_filename
        # )
        #
        # with open(self.vehicle_file, 'r') as f:
        #     self.vehicles = json.load(f)

    def getFutureVehicle(self, grid, vid_now, vid_before):
        # search vehicle information of vehicle list
        grid = grid.reshape((grid.shape[0], -1, grid.shape[-1]))


