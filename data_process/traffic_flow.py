#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : traffic_flow.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/5/13 15:35

import json
import os
import time
from config import *


# —————— Get the traffic flow real time ——————
class TrafficFlow(object):

    def __init__(self):
        self.config = Config()
        self.traffic_flow_file = os.path.join(self.config.path.data_path, self.config.traffic_filename)

        with open(self.traffic_flow_file, 'r') as f:
            self.traffic = json.load(f)

    def get_traffic_flow(self, t, road):
        return self.traffic[t][road]['vehicle_number']


if __name__ == '__main__':
    traffic = TrafficFlow()
    res = traffic.get_traffic_flow(11, '-gneE24')
    print(res)
