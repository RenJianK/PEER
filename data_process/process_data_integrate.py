#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : process_data.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/5/11 14:53

import numpy as np
import math
import random
from data_process.prepare_data import *
from data_process.traffic_flow import TrafficFlow
from data_process.traffic_light import TrafficLight

# If Not Necessary, the Substantiality will not be Added

# —————— Transfer data to input and output format ——————
from tool import judge_road_junction, get_lane_number

LineOfSight = [20, 40, 60, 80, 100]


def check_visible(center_v, center_h, target_v, target_h, g_m):
    # In order to best check the visibility of the vehicle, we connect
    # a line between the vehicle equipped with OBU  and the target
    # vehicle and check if there are vehicles along the route.
    angle = (target_v - center_v, target_h - center_h)
    sqrt_len = math.sqrt(angle[0] ** 2 + angle[1] ** 2)
    angle = (angle[0] / sqrt_len, angle[1] / sqrt_len)
    position_x, position_y = center_v, center_h

    while True:
        position_x += angle[0]
        position_y += angle[1]

        if round(position_x) == target_v and round(position_y) == target_h:
            return True

        if g_m[round(position_y)][round(position_x)][3] != 0:
            return False


def number_of_v2x(vehicles):
    n = 0
    for vehicle in vehicles:
        if vehicles[vehicle]['info']['is_obu'] == 1:
            n += 1
    return n


class ProcessData(object):
    """Process original data for learning model.

    A road will be divided into N x M grids, every grid with the same size, namely
    width x length, where width is the width of lane and length is the length of
    minimal vehicle's length add the safe gap. The grid will save the vehicle state
    if the vehicle locate in the grid, otherwise only fill with itself information.
    In addition, the input for the model will just contain vehicles that with obu or
    in the line of sight of V2X.

    Args:
        line_of_sight (float): The line of sight of sense in the vehicle with obu, it
            is assumed that the line of sight is a circle.
        max_length (int): In order to fix the shape of grid map, we fix the maximal length.
        max_lane (int): Fixing the lane number as above.
    """

    def __init__(self, max_length, max_lane):
        self.config = Config()  # basic configuration
        self.original_data = PrepareData(load=True)  # original data class
        self.line_of_sight = 0  # line of sight of vehicle equipped with obu
        self.traffic_light = TrafficLight()  # traffic light
        self.traffic_flow = TrafficFlow()  # traffic flow
        self.grid_length = self.config.min_gap + self.config.min_length  # grid length
        self.max_length = max_length  # maximal length
        self.max_lane = max_lane  # maximal lane

        self.v2x_train_data_file = os.path.join(
            self.config.path.data_path, self.config.v2x_train_data
        )
        self.v2x_test_data_file = os.path.join(
            self.config.path.data_path, self.config.v2x_test_data
        )
        self.v2x_data = os.path.join(
            self.config.path.data_path, self.config.v2x_data
        )

    def save_data(self, data):
        # random.shuffle(data)
        # print(len(data))
        # train_data = data[:int(len(data) * 0.8)]
        # test_data = data[int(len(data) * 0.8):]
        # with open(self.v2x_train_data_file, 'w') as f:
        #     json.dump(train_data, f)
        #
        # with open(self.v2x_test_data_file, 'w') as f:
        #     json.dump(test_data, f)

        with open(self.v2x_data, 'w') as f:
            json.dump(data, f)

    def process_data(self):
        # generate grid map of line of sight for v2x and the global grid map
        vehicle_file = os.path.join(
            self.config.path.data_path,
            self.config.vehicle_filename
        )

        with open(vehicle_file, 'r') as f:
            all_information = json.load(f)

        data = {}
        for distance in LineOfSight:
            self.line_of_sight = distance
            v2x_dataset = []

            i = 0
            for t, val in enumerate(all_information):
                # compute the global grid map
                i += 1
                if i % 100 == 0:
                    print("data process step {} | {}".format(i, len(all_information)))
                roads = val['vehicles']
                v2x_data = {}

                # from the origination to the destination
                for road in roads:
                    # eliminate the intersection
                    if not judge_road_junction(
                            road) or road in self.config.unfocused_list or road not in self.config.focused_list:
                        continue
                    vehicles = roads[road]['vehicles']
                    number = number_of_v2x(vehicles)
                    label = 1
                    if number < 1:
                        label = 0

                    g_m = self.generate_global_map(road, vehicles)
                    v_m = self.generate_v2x_map(t, road, vehicles, g_m)
                    v2x_data[road] = {
                        'global': g_m,
                        'v2x': v_m,
                        'useful': label
                    }

                v2x_dataset.append(v2x_data)

            data[distance] = v2x_dataset

        self.save_data(data)

    def generate_global_map(self, road, vehicles):
        # Generate the global grid map, namely, add all vehicles
        # into the grid belonging to itself. [x, y, s]

        l, n = self.get_road_shape(road)

        v_number, h_number = int(self.max_lane), math.ceil(self.max_length / self.grid_length)
        l_number = math.ceil(l / self.grid_length)
        m = np.zeros((h_number, v_number, 4))
        if v_number > n:
            m[:, n:][:] = -1

        if self.max_length > l:
            m[l_number:, :][:] = -1

        for vehicle in vehicles:
            info = vehicles[vehicle]['info']
            speed = info['speed']
            lane = get_lane_number(info['lane'])
            position_in_lane = info['position_in_lane']

            if math.floor(position_in_lane / self.grid_length) >= h_number:
                continue
            v_p = math.floor(position_in_lane / self.grid_length)

            m[v_p, lane] = [position_in_lane, lane, speed, 1]

        return m.tolist()

    def generate_v2x_map(self, t, road, vehicles, g_m):
        # Generate v2x map, which contains [x, y, s, c,
        # e, p, a, length-x, l, f]
        l, n = self.get_road_shape(road)

        v_number, h_number = int(self.max_lane), math.ceil(self.max_length / self.grid_length)
        l_number = math.ceil(l / self.grid_length)
        m = np.zeros((h_number, v_number, 10))

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
                # m[i, j] = [x, y, 0, 0, 1, 0, l, f]
                m[i, j] = [x, y, 0, 0, 1, 0, 0, self.config.max_road_length - x, l, f]

        # get vehicles information in the line of sight of v2x vehicles
        # assuming the line of sight is circle

        # 1. find the vehicles equipped with OBU to incline the search space
        v2x_list = []
        for vehicle in vehicles:
            info = vehicles[vehicle]['info']
            if info['is_obu'] == 1:
                speed = info['speed']
                lane = get_lane_number(info['lane'])
                position_in_lane = info['position_in_lane']
                accelerate = info['accelerate']

                h_p = math.floor(position_in_lane / self.grid_length)
                h_p = h_p if h_p < h_number else h_number - 1

                # m[h_p, lane][:6] = [position_in_lane, lane, speed, 1, 0, 0, accelerate]
                m[h_p, lane][:8] = [position_in_lane, lane, speed, 1, 0, 0, accelerate,
                                    self.config.max_road_length - position_in_lane]
                v2x_list.append([lane, position_in_lane, h_p])

        # 2. fill all visible section
        for v2x in v2x_list:
            grid_area = math.floor(
                self.line_of_sight /
                self.grid_length)  # the area of LOS
            center_v, center_h = v2x[0], v2x[2]
            left_grid = center_v - grid_area if (center_v - grid_area) >= 0 else 0
            right_grid = center_v + grid_area if (center_v + grid_area) <= int(n) - 1 else int(n) - 1
            backward_grid = center_h - grid_area if (center_h - grid_area) >= 0 else 0
            forward_grid = center_h + grid_area if (center_h + grid_area) <= l_number - 1 else l_number - 1

            # fill visible vehicle
            for j in range(left_grid, right_grid + 1):
                for i in range(backward_grid, forward_grid + 1):
                    if i == center_h and j == center_v:
                        continue
                    if g_m[i][j][3] != 0:
                        visibility = check_visible(center_v, center_h, j, i, g_m)
                        g = [g_m[i][j][0], g_m[i][j][1], g_m[i][j][2], 1, 0, 0] if visibility else m[i, j][:6]
                        m[i, j][:6] = g
                    else:
                        visibility = check_visible(center_v, center_h, j, i, g_m)
                        m[i, j][4] = 0 if visibility else 1

        return m.tolist()

    def get_road_shape(self, road_name):
        # get the shape of current road
        assert road_name in self.traffic_light.road_map, 'The road_name is error! Please check it!'

        road_info = self.traffic_light.road_map.get(road_name)
        length = road_info['length']
        lane_number = road_info['lane_num']

        return length, lane_number


if __name__ == '__main__':
    # 20, 30, 50, 80, 100
    process_data = ProcessData(300, 3)
    process_data.process_data()
