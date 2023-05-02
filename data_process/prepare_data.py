#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : prepare_data.py
# @Project : KDGAN-CSR
# @Author : RenJianK
# @Time : 2022/5/9 20:20

from config import *
import json
import sys
import traci
import random
from sumolib import checkBinary
import os


# ---- prepare original data in sumo ----

def initial_traci(sumocfg_file, if_show_gui=True):
    """Open the sumo platform

    Args:
        sumocfg_file: Absolute path of sumocfg file.
        if_show_gui: Whether show the gui when run the sumo platform.

    Returns:

    """
    assert 'SUMO_HOME' in os.environ, "please declare environment variable 'SUMO_HOME'"

    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

    sumo_binary = checkBinary(
        'sumo') if not if_show_gui else checkBinary('sumo-gui')

    traci.start([sumo_binary, '-c', sumocfg_file])


class PrepareData(object):
    """Prepare data for our work.


    Args:
       sumocfg_filename (string): Filename of .sumocfg file.
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
                        'route': the route of vehicle,
                        'lateral_position': lateral position in road,
                        'lateral_speed': lateral speed,
                        'angle': angle of vehicle's direction
                        }
                    }}
                }
            }
            ]
        __traffic_light (bool): Save traffic light information.
        __road_map (bool): Save the road map.
    """

    def __init__(self, sumocfg_filename=None, obu_rate=0.2, load=False):
        self.__config = Config()
        if not load:
            sumocfg_file = os.path.join(
                self.__config.path.sumo_path,
                sumocfg_filename)
            initial_traci(sumocfg_file)

        self.obu_rate = obu_rate
        # mark whether the vehicle is equipped with obu equipment, 0 indicate
        # with obu otherwise without
        self.label = {}

        self.__vehicles = []
        self.__roads_info = []
        self.__traffic_light = []
        self.__road_map = {}

        self.__roads_traffic_file = os.path.join(
            self.__config.path.data_path,
            self.__config.traffic_filename)
        self.__road_map_file = os.path.join(
            self.__config.path.data_path,
            self.__config.roadMap_filename)
        self.__traffic_light_file = os.path.join(
            self.__config.path.data_path,
            self.__config.trafficLight_filename)
        self.__vehicles_file = os.path.join(
            self.__config.path.data_path,
            self.__config.vehicle_filename)

        if not load:
            self.get_data()
        else:
            self.load_data()

    def get_train_vehicles(self):
        # all vehicles information in road network.
        roads = traci.edge.getIDList()
        times = traci.simulation.getTime()  # current time step
        vehicles_now = {}

        for road in roads:
            vehicles = traci.edge.getLastStepVehicleIDs(road)
            vehicles_in_road = {}
            for vehicle in vehicles:

                # when vehicle enters road network, we initialize
                # this vehicle whether is equipped with obu by
                # a deterministic probability
                if vehicle not in self.label:
                    self.label[vehicle] = 1 if random.random(
                    ) < self.obu_rate else 0

                objective_vehicle = {
                    'is_obu': self.label[vehicle],
                    'position': traci.vehicle.getPosition(vehicle),
                    'position_in_lane': traci.vehicle.getLanePosition(vehicle),
                    'speed': traci.vehicle.getSpeed(vehicle),
                    'lane': traci.vehicle.getLaneID(vehicle),
                    'accelerate': traci.vehicle.getAcceleration(vehicle),
                }
                vehicles_in_road[vehicle] = {
                    'id': vehicle,
                    'info': objective_vehicle
                }
            vehicles_now[road] = {
                'vehicles': vehicles_in_road,
                'road_id': road
            }
        self.__vehicles.append({
            'time': times,
            'vehicles': vehicles_now
        })

    def get_train_road(self):
        # get the traffic flow information of every road
        roads = traci.edge.getIDList()  # road list

        now = traci.simulation.getTime()

        road_info = {}

        for road in roads:
            # traversal all road and save their traffic flow or speed
            vehicle_number = traci.edge.getLastStepVehicleNumber(road)
            lane_number = traci.edge.getLaneNumber(road)
            vehicles_id = traci.edge.getLastStepVehicleIDs(road)
            length = traci.lane.getLength("{}_{}".format(road, 0))

            halt_number = traci.edge.getLastStepHaltingNumber(road)
            mean_speed = traci.edge.getLastStepMeanSpeed(road)
            road_info[road] = {
                'road_id': road,
                'vehicles': vehicles_id,
                'vehicle_number': vehicle_number,
                'lane_number': lane_number,
                'length': length,
                'halt_number': halt_number,
                'mean_speed': mean_speed,
            }

        self.__roads_info.append(road_info)

        if now == 1550:
            # save road map, it should be note that the road map only be
            # generated once to reduce redundant time
            for road in roads:
                lane_number = traci.edge.getLaneNumber(road)
                lanes = ["{}_{}".format(road, key)
                         for key in range(lane_number)]
                length = traci.lane.getLength("{}_{}".format(road, 0))
                lanes_in_road = []
                for lane in lanes:
                    shape = traci.lane.getShape(lane)
                    links = traci.lane.getLinks(lane)
                    if len(links) == 0:
                        # if some road has no neighbor, then the link-state set
                        # to None
                        lanes_in_road.append({
                            'lane_id': lane,
                            'next_lane_id': None,
                            'next_junction_id': None,
                            'next_turn_direction': None,
                            'next_junction_length': None,
                            'lane_shape': shape,
                            'width': traci.lane.getWidth(lane)
                        })
                        continue
                    else:
                        # if some road has neighbor, then its link-state is its
                        # neighbor
                        links = links[0]
                    next_lane_id = links[0]
                    next_junction_id = links[4]
                    next_turn_direction = links[6]
                    next_junction_length = links[-1]
                    lanes_in_road.append({
                        'lane_id': lane,
                        'next_lane_id': next_lane_id,
                        'next_junction_id': next_junction_id,
                        'next_turn_direction': next_turn_direction,
                        'next_junction_length': next_junction_length,
                        'lane_shape': shape,
                        'width': traci.lane.getWidth(lane)
                    })
                self.__road_map[road] = {
                    'lane_links': lanes_in_road,  # link state
                    'length': length,  # length of this road
                    'lane_num': lane_number  # the number of road
                }

    def get_train_light(self):
        traffic_list = traci.trafficlight.getIDList()  # traffic signal list
        now = traci.simulation.getTime()

        traffic_state_now = {}
        for light in traffic_list:
            # get every signal state
            tls_state = traci.trafficlight.getRedYellowGreenState(light)

            next_switch = traci.trafficlight.getNextSwitch(light)

            tls_phrase = traci.trafficlight.getPhase(light)
            tls_phrase_duration = traci.trafficlight.getPhaseDuration(light)
            controlled_lanes = traci.trafficlight.getControlledLanes(light)
            controlled_links = traci.trafficlight.getControlledLinks(light)

            traffic_state_now[light] = {
                'tls_state': tls_state,
                'next_switch': next_switch,
                'tls_now_phrase': tls_phrase,
                'tls_phrase_duration': tls_phrase_duration,
                'controlled_lanes': controlled_lanes,
                'controlled_links': controlled_links,
            }

        traffic_state_now['now'] = now
        self.__traffic_light.append(traffic_state_now)

    def get_train_data(self):
        self.get_train_vehicles()

        self.get_train_road()

        self.get_train_light()

    def save_data(self):
        # save data of our model
        with open(self.__roads_traffic_file, 'w') as f:
            json.dump(self.__roads_info, f)

        with open(self.__road_map_file, 'w') as f:
            json.dump(self.__road_map, f)

        with open(self.__traffic_light_file, 'w') as f:
            json.dump(self.__traffic_light, f)

        with open(self.__vehicles_file, 'w') as f:
            json.dump(self.__vehicles, f)

    def load_data(self):
        # load data of our model
        with open(self.__roads_traffic_file, 'r') as f:
            self.__roads_info = json.load(f)

        with open(self.__road_map_file, 'r') as f:
            self.__road_map = json.load(f)

        with open(self.__traffic_light_file, 'r') as f:
            self.__traffic_light = json.load(f)

        with open(self.__vehicles_file, 'r') as f:
            self.__vehicles = json.load(f)

    def get_data(self):
        # generate the dataset from sumo
        start = 1000
        end = 4000
        for step in range(start, end):
            traci.simulationStep(step)

            if (step - start) % 100 == 0:
                print(
                    "simulation step is {} | {}".format(
                        (step - start),
                        (end - start)))

            self.get_train_data()

        print("simulation load complete!")
        self.save_data()


if __name__ == '__main__':
    prepare_data = PrepareData('demo.sumocfg', load=False, obu_rate=0.1)
