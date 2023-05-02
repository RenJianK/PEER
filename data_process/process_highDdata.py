#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : process_highDdata.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/11/7 16:09

import numpy as np
import math
import random
from data_process.prepare_highD import *
from data_process.traffic_flow import TrafficFlow


# —————— Transfer data to input and output format ——————


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

    def __init__(self, line_of_sight, max_length, max_lane):
        self.config = Config()  # basic configuration
        self.original_data = PrepareData(load=True)  # original data class
        self.line_of_sight = line_of_sight  # line of sight of vehicle equipped with obu
        self.traffic_flow = TrafficFlow()  # traffic flow
        self.grid_length = self.config.min_gap + self.config.min_length  # grid length
        self.max_length = max_length  # maximal length
        self.max_lane = max_lane  # maximal lane

        self.v2x_data = os.path.join(
            self.config.path.data_path, self.config.v2x_data
        )

    def save_data(self, data):
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
                vehicles = roads[road]['vehicles']
                number = number_of_v2x(vehicles)
                label = 1
                if number < 1:
                    label = 0

                g_m = self.generate_global_map(t, road, vehicles)
                v_m = self.generate_v2x_map(t, road, vehicles, g_m)
                g_m = g_m[:, :, :-1]
                g_m = g_m.tolist()
                v2x_data[road] = {
                    'global': g_m,
                    'v2x': v_m,
                    'useful': label
                }

            v2x_dataset.append(v2x_data)
        self.save_data(v2x_dataset)

    def generate_global_map(self, t, road, vehicles):
        # Generate the global grid map, namely, add all vehicles
        # into the grid belonging to itself. [x, y, s]

        l, n = self.max_length, self.max_lane

        v_number, h_number = int(self.max_lane), math.ceil(self.max_length / self.grid_length)
        l_number = math.ceil(l / self.grid_length)
        m = np.zeros((h_number, v_number, 6))
        if v_number > n:
            m[:, n:][:] = -1

        if self.max_length > l:
            m[l_number:, :][:] = -1

        # vehicle will occupy multi grid if it's length
        # is longer than single grid
        for vehicle in vehicles:
            info = vehicles[vehicle]['info']
            speed = info['speed']
            lane = int(info['lane'])
            position_in_lane = info['position_in_lane']
            length = info['length']
            y = info['lateral_position']

            if math.floor(position_in_lane / self.grid_length) >= h_number:
                continue
            v_p = math.floor(position_in_lane / self.grid_length)

            occupied_grid_number = 1 + length // self.grid_length
            if occupied_grid_number % 2 != 0:
                start_grid = v_p - occupied_grid_number // 2
                end_grid = v_p + occupied_grid_number // 2 + 1
            else:
                start_grid = v_p - occupied_grid_number // 2 if position_in_lane % self.grid_length > (
                        self.grid_length // 2) else v_p + 1 - occupied_grid_number // 2
                end_grid = start_grid + occupied_grid_number

            start_grid, end_grid = int(start_grid), int(end_grid)
            end_grid = end_grid if end_grid < h_number else h_number
            start_grid = start_grid if start_grid > 0 else 0

            for index in range(start_grid, end_grid):
                position = position_in_lane - (v_p - index) * self.grid_length
                if occupied_grid_number == 1:
                    m[index, lane] = [position, y, lane, speed, 1, 0]
                else:
                    m[index, lane] = [position, y, lane, speed, 1, 1]

        # f = self.traffic_flow.get_traffic_flow(t, road)
        # b = np.sum(m[:, :, 4])
        # if f != b:
        #     print("啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊！")
        #     print(f, b)
        # print(f, b)
        return m

    def generate_v2x_map(self, t, road, vehicles, g_m):
        # Generate v2x map, which contains [x, y, lane, s, c,
        # e, p, a, length-x, length, f], which respectively means
        # coordinates, lane, speed, if car, if in UPAs, if need repainted
        # accelerate, length to terminal, flow, length
        l, n = self.max_length, self.max_lane
        road_shape = self.original_data.get_road_shape(road)

        v_number, h_number = int(self.max_lane), math.ceil(self.max_length / self.grid_length)
        l_number = math.ceil(l / self.grid_length)
        m = np.zeros((h_number, v_number, 11))

        if v_number > n:
            m[:, n:][:] = -1

        if self.max_length > l:
            m[l_number:, :][:] = -1

        # initialize grid map
        for i in range(h_number):
            for j in range(v_number):
                x, lane = i * self.grid_length, j
                y = (road_shape[j] + road_shape[j + 1]) / 2
                f = self.traffic_flow.get_traffic_flow(t, road)
                # m[i, j] = [x, y, 0, 0, 1, 0, f]
                m[i, j] = [x, y, lane, 0, 0, 1, 0, 0, self.config.max_road_length - x, self.grid_length, f]

        # get vehicles information in the line of sight of v2x vehicles
        # assuming the line of sight is circle

        # 1. find the vehicles equipped with OBU to incline the search space
        v2x_list = []
        for vehicle in vehicles:
            info = vehicles[vehicle]['info']
            if info['is_obu'] == 1:
                speed = info['speed']
                lane = int(info['lane'])
                position_in_lane = info['position_in_lane']
                accelerate = info['acceleration']
                length = info['length']
                y = info['lateral_position']

                h_p = math.floor(position_in_lane / self.grid_length)
                h_p = h_p if h_p < h_number else h_number - 1

                occupied_grid_number = 1 + length // self.grid_length
                if occupied_grid_number % 2 != 0:
                    start_grid = h_p - occupied_grid_number // 2
                    end_grid = h_p + occupied_grid_number // 2 + 1
                else:
                    start_grid = h_p - occupied_grid_number // 2 if position_in_lane % self.grid_length > (
                            self.grid_length // 2) else h_p + 1 - occupied_grid_number // 2
                    end_grid = start_grid + occupied_grid_number

                start_grid, end_grid = int(start_grid), int(end_grid)
                end_grid = end_grid if end_grid < h_number else h_number
                start_grid = start_grid if start_grid > 0 else 0

                for index in range(start_grid, end_grid):
                    position = position_in_lane - (h_p - index) * self.grid_length
                    m[index, lane][:10] = [position, y, lane, speed, 1, 0, 0, accelerate,
                                          self.config.max_road_length - position_in_lane, length]
                    v2x_list.append([lane, position, h_p])

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

                    # if the length of target is longer than grid,
                    # it will occupy many grids
                    if g_m[i, j][4] != 0:
                        visibility = check_visible(center_v, center_h, j, i, g_m)
                        if g_m[i, j][-1] == 1:
                            index = i
                            while index < h_number and g_m[index, j][-1] == 1:
                                g = [g_m[index, j][0], g_m[index, j][1], g_m[index, j][2], g_m[index, j][3], 1, 0, 0] if \
                                    visibility else m[index, j][:7]
                                m[index, j][:7] = g
                                index += 1
                        else:
                            g = [g_m[i, j][0], g_m[i, j][1], g_m[i, j][2], g_m[i, j][3], 1, 0, 0] if visibility else m[i, j][:7]
                            m[i, j][:7] = g
                    else:
                        visibility = check_visible(center_v, center_h, j, i, g_m)
                        m[i, j][5] = 0 if visibility else 1

        return m.tolist()


if __name__ == '__main__':
    process_data = ProcessData(100, 300, 3)
    process_data.process_data()
