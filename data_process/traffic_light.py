#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : traffic_light.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/5/12 15:49

import json
import os
from config import *


class TrafficLight(object):
    """Search the traffic light state.

    The light state of signal lane can be find by this class.
    """

    def __init__(self):
        self.config = Config()
        self.traffic_light_file = os.path.join(
            self.config.path.data_path,
            self.config.trafficLight_filename)
        self.road_map_file = os.path.join(
            self.config.path.data_path,
            self.config.roadMap_filename)

        with open(self.traffic_light_file, 'r') as f:
            self.light = json.load(f)

        with open(self.road_map_file, 'r') as f:
            self.road_map = json.load(f)

    def get_current_state(self, time, lane_name):
        road_id = lane_name.split('_')[0]
        junction_lane_name = None
        for val in self.road_map:
            if val == road_id:
                junction_lane_name = self.road_map[val]['lane_links'][0]["next_junction_id"]
                break

        assert junction_lane_name is not None, 'the lane name entered is not in the correct format'

        light = self.light[time]
        junction_name = junction_lane_name.split('_')[0]
        junction_name = junction_name[1:]

        tls_state = light[junction_name]['tls_state']
        controlled_lanes = light[junction_name]['controlled_lanes']
        # print(controlled_lanes)
        # print(lane_name)
        # print(tls_state)
        # print(junction_name)
        state = tls_state[controlled_lanes.index(lane_name)]

        return 1 if state == 'g' or state == 'G' else 0


if __name__ == '__main__':
    traffic_light = TrafficLight()
    light = traffic_light.get_current_state(2, "-gneE24_0")
    print(light)
