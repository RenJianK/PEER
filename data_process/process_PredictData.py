#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : process_PredictData.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/9/3 17:00
import json
import math
import os
import numpy as np

from config import Config
from data_process.prepare_data import PrepareData
from data_process.traffic_flow import TrafficFlow
from data_process.traffic_light import TrafficLight
from tool import judge_road_junction, get_lane_number


class PredictData(object):
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
        self.original_data = PrepareData(load=True)
        self.traffic_light = TrafficLight()  # traffic light
        self.traffic_flow = TrafficFlow()  # traffic flow
        self.grid_length = self.config.min_gap + self.config.min_length  # grid length
        self.max_length = self.config.max_road_length  # maximal length
        self.max_lane = self.config.max_lane_num  # maximal lane

        self.predict_train_data_file = os.path.join(
            self.config.path.data_path, self.config.predict_train_data_path
        )

    def save(self, data):
        with open(self.predict_train_data_file, 'w') as f:
            json.dump(data, f)

    def process_data(self):
        # generate the train data of the group trajectory predict model
        # the construction of data is [
        # [
        #    "road_name": {
        #       "grid": grid,
        #       "vid": vid
        #    }
        # ], ...]

        # 1. First, load the road state data
        vehicle_file = os.path.join(
            self.config.path.data_path,
            self.config.vehicle_filename
        )
        with open(vehicle_file, 'r') as f:
            all_information = json.load(f)

        # 2. Second, initial the data storage variable
        data = []

        # 3. Third, traverse data
        i = 0
        for t, val in enumerate(all_information):
            # compute the global grid map
            i += 1
            if i % 100 == 0:
                print("data process step {} | {}".format(i, len(all_information)))

            instantaneous = {}
            roads = val['vehicles']

            for road in roads:
                # eliminate the intersection
                if not judge_road_junction(
                        road) or road in self.config.unfocused_list or road not in self.config.focused_list:
                    continue

                vehicles = roads[road]['vehicles']
                road_data = self.grid_map(t, road, vehicles)

                instantaneous[road] = road_data

            data.append(instantaneous)

        self.save(data)

    def grid_map(self, t, road, vehicles):
        # Generate grid map, which contains [x, y, s, c, a, l, f]
        l, n = self.get_road_shape(road)

        v_number, h_number = int(self.max_lane), math.ceil(self.max_length / self.grid_length)
        l_number = math.ceil(l / self.grid_length)
        m = np.zeros((h_number, v_number, 8))  # storage grid information
        v = ['' for _ in range(h_number * v_number)]  # storage the vehicle id

        if v_number > n:
            m[:, n:][:] = -1

        if self.max_length > l:
            m[l_number:, :][:] = -1

        # initialize grid map
        for i in range(h_number):
            for j in range(v_number):
                x, y = i * self.grid_length, j
                lane_name = '{}_{}'.format(road, j)
                l, f = self.traffic_flow.get_traffic_flow(
                    t, road), self.traffic_light.get_current_state(
                    t, lane_name)
                m[i, j] = [x, y, 0, 0, 0, self.config.max_road_length - x, l, f]
                # m[i, j] = [x, y, 0, 0, 0, l, f]

        # fill the vehicle state into the grid occupied by vehicle
        for vehicle in vehicles:
            info = vehicles[vehicle]['info']
            speed = info['speed']
            lane = get_lane_number(info['lane'])
            position_in_lane = info['position_in_lane']
            accelerate = info['accelerate']

            h_p = math.floor(position_in_lane / self.grid_length)
            h_p = h_p if h_p < h_number else h_number - 1

            m[h_p, lane][:6] = [position_in_lane, lane, speed, 1, accelerate, self.config.max_road_length - position_in_lane]
            # m[h_p, lane][:4] = [position_in_lane, lane, speed, 100]
            # print(h_p * v_number + lane)
            # print(len(v))

            v[h_p * v_number + lane] = vehicle

        return {
            "grid": m.tolist(),
            "vid": v
        }

    def get_road_shape(self, road_name):
        # get the shape of current road
        assert road_name in self.traffic_light.road_map, 'The road_name is error! Please check it!'

        road_info = self.traffic_light.road_map.get(road_name)
        length = road_info['length']
        lane_number = road_info['lane_num']

        return length, lane_number


if __name__ == '__main__':
    pre = PredictData()
    pre.process_data()
