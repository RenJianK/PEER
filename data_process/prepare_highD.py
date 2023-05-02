#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : prepare_highD.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/11/5 19:27
import pandas

from config import *
import json
import random
import os

# ---- prepare original data in highD datasets ----
LENGTH = 300
TIME_SCALE = 5


class PrepareData(object):
    """Prepare data for our work.


    Args:
       highD_filename (string): Filename of .highD directory.
       obu_rate (float):  Proportion of vehicle with obu, default 0.3.

    Attributes:
        __config (Config): See
            :class: `config.Config`.
        obu_rate (float): See above.
        label (float): See follows.
        __vehicles (list): Save the vehicle state.
            The architecture is: [
            {
            'time': time,
            'vehicles': {
                'road_id': $road_id,
                $road_id: {
                $id: {
                    'id': id,
                    'info': {
                        'is_obu': 1 or 0,
                        'position': coordinate in 2D,
                        'position_in_lane': position in lane,
                        'speed': speed,
                        'lane': lane number,
                        'road': road_number,
                        'length': vehicle length,
                        'width': vehicle width,
                        'accelerate': accelerate,
                        }
                    }}
                }
            }
            ]
        __road_map (bool): Save the road map.
    """

    def __init__(self, highD_filename='data', obu_rate=0.5, load=False):
        self.__config = Config()
        self.highD_file = os.path.join(
            self.__config.path.highD_path,
            highD_filename
        )

        self.obu_rate = obu_rate
        self.grid_length = self.__config.min_length + self.__config.min_gap
        # mark whether the vehicle is equipped with obu equipment, 0 indicate
        # with obu otherwise without
        self.label = {}

        self.__vehicles = []
        self.__roads_info = []
        self.__roads_state = {}

        self.__roads_traffic_file = os.path.join(
            self.__config.path.data_path,
            self.__config.traffic_filename)
        self.__roads_state_file = os.path.join(
            self.__config.path.data_path,
            self.__config.roads_state
        )
        self.__vehicles_file = os.path.join(
            self.__config.path.data_path,
            self.__config.vehicle_filename)

        if not load:
            self.get_data()
        else:
            self.load_data()

    def save_data(self):
        # save data of our model
        with open(self.__roads_traffic_file, 'w') as f:
            json.dump(self.__roads_info, f)

        with open(self.__vehicles_file, 'w') as f:
            json.dump(self.__vehicles, f)

        with open(self.__roads_state_file, 'w') as f:
            json.dump(self.__roads_state, f)

    def load_data(self):
        # load data of our model
        with open(self.__roads_traffic_file, 'r') as f:
            self.__roads_info = json.load(f)

        with open(self.__vehicles_file, 'r') as f:
            self.__vehicles = json.load(f)

        with open(self.__roads_state_file, 'r') as f:
            self.__roads_state = json.load(f)

    def get_data(self):
        # generate the dataset from sumo
        end = 6000
        information = {}
        road_info = {}
        # 首先按照时间获取状态
        for i in range(1, 60):
            meta_filename = '0{}_{}'.format(i, self.__config.meta_suffix) if i < 10 else \
                '{}_{}'.format(i, self.__config.meta_suffix)
            filename = '0{}_{}'.format(i, self.__config.tracks_suffix) if i < 10 else \
                '{}_{}'.format(i, self.__config.tracks_suffix)
            meta_filename = os.path.join(self.highD_file, meta_filename)
            tracks_filename = os.path.join(self.highD_file, filename)

            if not (os.path.isfile(meta_filename) and os.path.isfile(tracks_filename)):
                continue

            tracks = pandas.read_csv(tracks_filename)
            laneNum = max(tracks.loc[:, 'laneId'])
            if laneNum != 8:
                continue

            meta = pandas.read_csv(meta_filename)
            upperLaneMarking = meta.loc[0, 'upperLaneMarkings']
            upperLaneMarking = list(map(lambda v: float(v), upperLaneMarking.split(';')))
            lowerLaneMarking = meta.loc[0, 'lowerLaneMarkings']
            lowerLaneMarking = list(map(lambda v: float(v), lowerLaneMarking.split(';')))
            vehicle_upper = []
            vehicle_lower = []
            road_info_upper = []
            road_info_lower = []
            upper = i * 2
            lower = i * 2 + 1
            # 按照时间存储车辆信息
            for j in range(1, end):
                if j % TIME_SCALE != 0:
                    continue
                # 需要按照车辆的上下行位置区分道路
                vehicles = tracks.loc[tracks.loc[:, 'frame'] == j]
                upper_road = {}
                lower_road = {}
                upper_vehicle_number = 0
                lower_vehicle_number = 0
                vehicle_id_upper = []
                vehicle_id_lower = []

                for k in range(vehicles.shape[0]):
                    vehicle = vehicles.iloc[k]
                    x, y = vehicle['x'], vehicle['y']
                    width, height = vehicle['width'], vehicle['height']
                    xVelocity, yVelocity = vehicle['xVelocity'], vehicle['yVelocity']
                    xAcceleration, yAcceleration = vehicle['xAcceleration'], vehicle['yAcceleration']
                    # 从行驶方向到路段尽头的距离以及从起始位置到车辆中心点的距离
                    frontSightDistance, backSightDistance = vehicle['frontSightDistance'], vehicle[
                        'backSightDistance']
                    laneId = vehicle['laneId']
                    laneId = laneId - 2 if laneId < 5 else 8 - laneId  # 0, 1, 2
                    xVelocity = abs(xVelocity)
                    is_lower = True if lowerLaneMarking[0] <= y <= lowerLaneMarking[-1] else False

                    if backSightDistance > LENGTH:
                        continue

                    width = width if (width / 2 + backSightDistance) <= LENGTH else width / 2 + (LENGTH - backSightDistance)
                    b = (width // self.grid_length) + 1 if width > self.grid_length else 1
                    b = int(b)

                    if is_lower:
                        yVelocity = - yVelocity
                        y = lowerLaneMarking[-1] - y
                        vehicle_id = '{}_{}'.format(lower, vehicle['id'])
                        lower_vehicle_number += b
                        for i in range(b):
                            vehicle_id_lower.append(vehicle_id)
                    else:
                        upper_vehicle_number += b
                        xAcceleration = -xAcceleration
                        y = y - upperLaneMarking[0]
                        vehicle_id = '{}_{}'.format(upper, vehicle['id'])
                        for i in range(b):
                            vehicle_id_upper.append(vehicle_id)

                    if vehicle_id not in self.label:
                        self.label[vehicle_id] = 1 if random.random(
                        ) < self.obu_rate else 0

                    objective_vehicle = {
                        'is_obu': self.label[vehicle_id],
                        'position_in_lane': backSightDistance,
                        'lateral_position': y,
                        'speed': xVelocity,
                        'lane': laneId,
                        'length': width,
                        'width': height,
                        'acceleration': xAcceleration,
                        'lateral_speed': yVelocity
                    }

                    if is_lower:
                        lower_road[vehicle_id] = {
                            'id': vehicle_id,
                            'info': objective_vehicle
                        }
                    else:
                        upper_road[vehicle_id] = {
                            'id': vehicle_id,
                            'info': objective_vehicle
                        }

                road_info_upper.append({
                    'road_id': upper,
                    "vehicles": vehicle_id_upper,
                    "vehicle_number": upper_vehicle_number,
                    "lane_number": 3,
                    "length": LENGTH
                })
                road_info_lower.append({
                    'road_id': lower,
                    "vehicles": vehicle_id_lower,
                    "vehicle_number": lower_vehicle_number,
                    "lane_number": 3,
                    "length": LENGTH
                })
                vehicle_upper.append(upper_road)
                vehicle_lower.append(lower_road)

            # 26.35, 30.09, 33.65, 37.56
            upper_shape = [val - upperLaneMarking[0] for val in upperLaneMarking]
            lower_shape = [lowerLaneMarking[-1] - val for val in lowerLaneMarking][::-1]
            self.__roads_state[upper] = upper_shape
            self.__roads_state[lower] = lower_shape

            information[upper] = {
                'vehicles': vehicle_upper,
                'road_id': upper
            }
            information[lower] = {
                'vehicles': vehicle_lower,
                'road_id': lower
            }
            road_info[upper] = road_info_upper
            road_info[lower] = road_info_lower

        res = []
        road_res = []
        for i in range((end // TIME_SCALE) - 2):
            res_t = {}
            road_t = {}
            for rid in information:
                res_t[rid] = {
                    "vehicles": information[rid]['vehicles'][i],
                    "road_id": rid
                }

            for rid in road_info:
                road_t[rid] = road_info[rid][i]

            res.append({
                "time": i,
                "vehicles": res_t
            })
            road_res.append(road_t)

        self.__vehicles = res
        self.__roads_info = road_res

        self.save_data()
        print("highD simulation load complete!")

    def get_road_shape(self, road_name):
        return self.__roads_state[road_name]

if __name__ == '__main__':
    prepare_data = PrepareData('data', load=False, obu_rate=0.5)
