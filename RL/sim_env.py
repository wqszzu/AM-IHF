"""Simulation environment of spatial crowdsourcing from client side."""
from __future__ import print_function
import argparse
import collections
import datetime
import glob
import logging
import math
import os
import time
import threading
from tkinter import _flatten

import numpy.random as random
from collections import deque
import re
import sys
import copy
import weakref
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal,Qt
from PyQt5.QtGui import QIcon,QFont
from PyQt5.QtChart import *

from PythonAPI.system.HCIS_main import *
import PythonAPI.util.globalvar as gl

import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q

#Find CARLA module
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Add PythonAPI for release mode
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
import re
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import vector

#Global functions
def find_weather_presets():
    """ Method to find weather preset"""

    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    '''
      Method to get actor display name
    :param actor:
    :param truncate:
    :return:
    '''

    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class Task(object):
    def __init__(self,index):
        self.index=index
        self.sim_actor=None
        self.sim_agent = None
        self.publish_time=None
        self.deadline=None
        self.payoff=1
        self.location=None

class Worker(object):
    def __init__(self,index):
        self.index=index
        self.sim_actor=None
        self.sim_agent=None
        self.current_task = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.hud=None
        self.cumulative_revenue=random.randint(1,50)
        self.speed_deque = deque(maxlen=50)
        self.complete_task_num=random.randint(1,30)


class BufferPool(object):
    def __init__(self):
        self.worker_list = []
        self.task_list = []

class RoadNetwork(object):
    def __init__(self):
        self.graph=None
        self.nodes=[]
        self.edges=[]
        self.zones=[]


class Zone(object):
    def __init__(self,index):
        self.index=index
        self.s_lon=0
        self.s_lat=0
        self.e_lon=0
        self.e_lat=0
        self.densities = 0
        self.worker_num=0
        self.task_num=0


class World(object):
    """ simulation world environment"""

    def __init__(self, carla_world, args,world_name=None):
        """Constructor method"""

        self._args = args
        self.world = carla_world
        self.sim_done=False

        self.grid_num = 6
        self.state_space=9*self.grid_num*self.grid_num
        self.action_space=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
        # the range of time window
        self.min_window = self._args.minWindow
        self.max_window = self._args.maxWindow

        #worker number
        self.worker_num=args.workerNum
        #worker set
        self.worker_list=[]
        #worker maximum speed(km/h)
        self.max_worker_speed=30

        #task number
        self.task_num=args.taskNum
        self.task_list=[]
        self.task_num_list=[]
        self.cumulative_task_num=0


        self.world_time=0
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self.hud = HUD(self._args.width,self._args.width)

        self.world_name = world_name
        self.player = None

        self.bufferPool = BufferPool()

    
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.restart(args)
        # self.world.on_tick(self.hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self._sampling_resolution = 2.0
        self.worker = self.worker_list[0]

     

        if self.world_name=="Town10HD_Opt":
            self.max_distance = 258.4828
            self.max_lon=110.0
            self.min_lon=-114.0
            self.max_lat=141.0
            self.min_lat=-69.0
        elif self.world_name == "Town01":
            self.max_distance = 246.5589
            self.max_lon = 396.0
            self.min_lon = -2.0
            self.max_lat = 331.0
            self.min_lat = -2.0
        elif self.world_name == "Town02":
            self.max_distance = 135.3717
            self.max_lon = 194.0
            self.min_lon = -7.0
            self.max_lat = 307.0
            self.min_lat = 105.0
        elif self.world_name=="Town03":
            self.max_distance = 408.8566
            self.max_lon = 246.0
            self.min_lon = -149.0
            self.max_lat = 208.0
            self.min_lat = -208.0
        elif self.world_name == "Town04":
            self.max_distance = 556.496
            self.max_lon = 414.0
            self.min_lon = -515.0
            self.max_lat = 352.0
            self.min_lat = -396.0
        elif self.world_name == "Town05":
            self.max_distance = 247.1178
            self.max_lon = 211.0
            self.min_lon = -276.0
            self.max_lat = 209.0
            self.min_lat = -208.0
        elif self.world_name == "Town06":
            self.max_distance = 377.5516
            self.max_lon = 673.0
            self.min_lon = -341.0
            self.max_lat = 362.0
            self.min_lat = -122.0
        elif self.world_name == "Town07":
            self.max_distance = 114.5364
            self.max_lon = 74.0
            self.min_lon = -203.0
            self.max_lat = 120.0
            self.min_lat = -249.0

        self.build_topology()
        self.build_graph()
        self.road_network=RoadNetwork()
        self.road_network.graph=self.graph
      
        self.road_network_parser()

      
        self.construct_road_zones()

        # Human-computer interaction screen
        self.hci = HCI(self._args.width, self._args.width, self)
        self.precipitation_deque = deque(maxlen=50)
        self.wind_intensity_deque = deque(maxlen=50)
        self.fog_density_deque = deque(maxlen=50)
        self.history_tasks_deque=deque(maxlen=50)

        self.total_train_step=0
        self.current_train_step=0
        self.max_time_window_deque=deque(maxlen=120)
        self.min_time_window_deque = deque(maxlen=120)
        self.current_time_window_deque=deque(maxlen=120)
        self.time_window_revenue_deque = deque(maxlen=500)
        self.loss_deque = deque(maxlen=500)

        self.total_reward=0
        self.total_gen_task_num=0
        self.total_com_task_num=0

    def get_max_distance(self):
        '''
        Get the current farthest distance in the world
        location_1.distance(location_2) can calculate the distance from loc1 to loc2
        :return:
        '''

        max_length=0
        points=self.map.get_spawn_points()
        for pos_1 in points:
            origin=pos_1.location
            for pos_2 in points:
                destination=pos_2.location
                route,length=self.worker_list[0].sim_agent.trace_route_length(origin, destination)
                if length>max_length:
                    max_length=length
        return max_length

    def reset_worker(self, worker):

        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        location = worker.sim_actor.get_transform().location
     
        worker.sim_actor.destroy()
        worker.sim_actor = None
        worker.sim_actor = None
        while worker.sim_actor is None:
            new_spawn_point = self.get_near_spawn_point(location)
            worker.sim_actor = self.world.try_spawn_actor(blueprint, new_spawn_point)

        self.modify_vehicle_physics(worker.sim_actor)
        if self._args.agent == "Basic":
            worker.sim_agent = BasicAgent(worker.sim_actor)
        else:
            worker.sim_agent = BehaviorAgent(worker.sim_actor, behavior=self._args.behavior)

        worker.collision_sensor.sensor.destroy()
        worker.collision_sensor = CollisionSensor(worker.sim_actor, worker.hud)

        if worker.current_task!=None:
            worker.sim_agent.set_destination(worker.current_task.location)

    def restart(self, args):
        """Restart the world"""


        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))

        worker_blueprints=self.world.get_blueprint_library().filter(self._actor_filter)
        worker_blueprint=random.choice(worker_blueprints)

        blueprint.set_attribute('role_name', 'NPC')

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if not self.map.get_spawn_points():
            print('There are no spawn points available in your map/town.')
            print('Please add some Vehicle Spawn Point to your UE4 scene.')
            sys.exit(1)
        spawn_points = self.map.get_spawn_points()

   
        for i in range(self.worker_num):

            worker_index="worker_"+str(i)
            worker=Worker(worker_index)
          
            worker_blueprint.set_attribute('role_name', worker_index)
            while worker.sim_actor is None:
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                worker.sim_actor=self.world.try_spawn_actor(worker_blueprint, spawn_point)
    
            self.modify_vehicle_physics(worker.sim_actor)

            # Set up the sensors.
            # Keep same camera config if the camera manager exists.
            cam_index = worker.camera_manager.index if worker.camera_manager is not None else 0
            cam_pos_id = worker.camera_manager.transform_index if worker.camera_manager is not None else 0
            worker.hud = HUD(self._args.width, self._args.height)
            worker.collision_sensor = CollisionSensor(worker.sim_actor, worker.hud)
            worker.lane_invasion_sensor = LaneInvasionSensor(worker.sim_actor, worker.hud)
            worker.gnss_sensor = GnssSensor(worker.sim_actor)
            worker.camera_manager = CameraManager(worker.sim_actor, worker.hud)
            worker.camera_manager.transform_index = cam_pos_id
            worker.camera_manager.set_sensor(cam_index, notify=False)
            actor_type = get_actor_display_name(worker.sim_actor)
            #get server fps
            self.world.on_tick(worker.hud.on_world_tick)

            worker.hud.notification(actor_type)
            self.worker_list.append(worker)



        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()



    def change_weather(self):
        '''
        Modify the humidity, fog concentration, and wind speed of the current weather
        :return:
        '''
        # Obtain precipitation, wind intensity and fog concentration from the simulation environment
        current_weather = self.world.get_weather()
        precipitation = current_weather.precipitation+random.randint(-5, 5)
        wind_intensity = current_weather.wind_intensity+random.randint(-5, 5)
        fog_density = current_weather.fog_density+random.randint(-5, 5)

        if(precipitation<0):
            precipitation=0
        if(wind_intensity<0):
            wind_intensity=0
        if(fog_density<0):
            fog_density=0

        new_weather = carla.WeatherParameters(
            precipitation=precipitation,
            wind_intensity=wind_intensity,
            fog_density=fog_density
        )

        self.world.set_weather(new_weather)

    def auto_gen_tasks(self,time_slice):
        '''
        Generate several tasks based on the current time
        :param time_slice: current time slice
        :return:
        '''
        num=random.randint(0,9)
        if num>3:
            self.gen_task()
            self.total_gen_task_num+=1


    def get_near_spawn_point(self,walker_location):
        '''
        Find the nearest spawn point based on pedestrian location
        :param worker_location: worker location
        :return: nearest point (including location and rotation)
        '''
        spawn_points = self.map.get_spawn_points()
        near_point=None
        min_distance=999999
        for point in spawn_points:
            distance=point.location.distance(walker_location)
            if distance<min_distance:
                min_distance=distance
                near_point=point

        if near_point==None:
            near_point= random.choice(spawn_points)

        return near_point


    def gen_task(self):
        '''
        Generate a task in the simulation environment
        :return:
        '''
        task_index="task_"+str(self.world_time)
        task=Task(task_index)
        task.publish_time=self.world_time
        self.waiting_time=random.randint(200,300)
        task.deadline=self.world_time+self.waiting_time
        task_blueprint = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        while task.sim_actor is None:
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            task.sim_actor = self.world.try_spawn_actor(task_blueprint, spawn_point)
        task.location=self.get_near_spawn_point(spawn_point.location).location
        self.bufferPool.task_list.append(task)
        self.cumulative_task_num+=1
        return task

    def update_buffer_pool(self):
        '''
        Refresh this buffer pool
        :return:
        '''
        #Refresh the current available workers
        for worker in self.worker_list:
            if worker.current_task!=None:
                if worker in self.bufferPool.worker_list:
                    self.bufferPool.worker_list.remove(worker)
            else:
                if  worker not in self.bufferPool.worker_list:
                    self.bufferPool.worker_list.append(worker)

        #Refresh all available tasks
        for task in self.bufferPool.task_list:
            if task.deadline<self.world_time:
                self.bufferPool.task_list.remove(task)
                task.sim_actor.destroy()
     

    def normalization(self, x, Max, Min):
        '''
        Data normalization
        :param x: the value to be normalized
        :param Max: Maximum value of normalized range
        :param Min: Minimum value of normalized range
        :return: normalized value
        '''
        x = (x - Min) / (Max - Min)
        return x

    def get_reward(self,worker,task):
        '''
        Calculate the reward for matching the current worker and task
        :param worker:
        :param task:
        :return:
        '''
        route, length = worker.sim_agent.trace_route_length(worker.sim_actor.get_transform().location, task.location)
        nor_cost=self.normalization(length,self.max_distance,0)
        reward=task.payoff-nor_cost
        return reward


    def greedy_match(self):
        '''
        greedy matching
        :param current_time_step: current time
        :return:Total revenue from matching in this round
        '''

        total_reward=0
        for task in self.bufferPool.task_list:
            max_reward=0
            max_worker=None
            for worker in self.bufferPool.worker_list:
                reward=self.get_reward(worker,task)
                if reward>max_reward:
                    max_reward=reward
                    max_worker=worker
            if max_worker!=None:
                max_worker.current_task=task
                #Calculate revenue
                max_worker.cumulative_revenue+=max_reward
                max_worker.complete_task_num+=1
                max_worker.sim_agent.set_destination(task.location)
                self.bufferPool.task_list.remove(task)
                self.bufferPool.worker_list.remove(max_worker)
                total_reward+=max_reward

        return total_reward

    def loction_to_zone(self, location):
        '''
        Get the zone where the current location is located
        :param location: current coordinate
        :return: the index of zone
        '''
        lon = location.x
        lat = location.y
        zone_x = int((lon - self.min_lon) / self.avg_lon)
        zone_y = int((lat - self.min_lat) / self.avg_lat)

        zone_index = zone_x * self.grid_num + zone_y
        if zone_index >= len(self.zones_list):
            zone_index = len(self.zones_list) - 1

        return zone_index

    def loction_to_grid(self, location):
        '''
        Get the grid where the current location is located
        :param location: current coordinate
        :return: grid_x, grid_y
        '''
        lon = location.x
        lat = location.y
        grid_x = int((lon - self.min_lon) / self.avg_lon)
        grid_y = int((lat - self.min_lat) / self.avg_lat)

        if grid_x >= self.grid_num:
            grid_x = self.grid_num-1

        if grid_y>=self.grid_num:
            grid_y=self.grid_num-1

        return grid_x, grid_y


    def get_route_length(self,worker):
        '''
        Calculate the distance between the worker's current position and the destination
        :param worker:
        :return:
        '''
        #Obtain the list of coordinate points of the worker's driving path
        points_list=worker.sim_agent.get_route()
        length=0
        start=worker.sim_actor.get_transform().location
        for point in points_list:
            end=point[0].transform.location
            distance=start.distance(end)
            length+=distance
            start=end
        return length

    def get_worker_remaining_time(self):
        remaining_time_list=[]
        total_time_list=[]
        for worker in self.worker_list:
            task=worker.current_task
            if task!=None:
                route, length = worker.sim_agent.trace_route_length(worker.sim_actor.get_transform().location,
                                                                    task.location)
                #length:km, speed:km/h 1h=3600s
                remaining_time=(length/self.max_worker_speed)*360
                remaining_time_list.append(remaining_time)
                total_time_list.append((self.world_time-task.publish_time)+remaining_time)
            else:
                remaining_time_list.append(0)
                total_time_list.append(1)

        return remaining_time_list,total_time_list

    def get_worker_remaining_time_grid(self):

        remaining_time_list = [[0 for i in range(self.grid_num)] for j in range(self.grid_num)]

        for worker in self.worker_list:
            try:
                task = worker.current_task
                if task != None:
                    route, length = worker.sim_agent.trace_route_length(worker.sim_actor.get_transform().location,
                                                                        task.location)
                    # length:km, speed:km/h 1h=3600s
                    remaining_time = (length / self.max_worker_speed) * 360
                    #Determine which grid the current location belongs to
                    grid_x,grid_y=self.loction_to_grid(worker.sim_actor.get_transform().location)

                    remaining_time_list[grid_x][grid_y]+=remaining_time
            except Exception as e:
                print("Time Error:", e)
                print("Error line:", e.__traceback__.tb_lineno)
                # self.reset_worker(worker)

        return remaining_time_list

    def get_task_deadline(self):
        deadline_list=[]
        for task in self.bufferPool.task_list:
            deadline=task.deadline-self.world_time
            if deadline<1:
                deadline=1
            deadline_list.append(deadline)

        return deadline_list

    def get_task_deadline_grid(self):

        deadline_list = [[999 for i in range(self.grid_num)] for j in range(self.grid_num)]
        for task in self.bufferPool.task_list:
            grid_x, grid_y = self.loction_to_grid(task.sim_actor.get_transform().location)
            try:
                deadline_list[grid_x][grid_y]+=task.deadline - self.world_time
            except:
                print("x:",grid_x)
                print("y:",grid_y)

        return deadline_list

    def get_worker_location(self):
        worker_location_list=[]
        for worker in self.worker_list:
            location=worker.sim_actor.get_transform().location
            worker_location_list.append([location.x,location.y])
        return worker_location_list

    def get_worker_location_grid(self):
        worker_location_list =  [[0 for i in range(self.grid_num)] for j in range(self.grid_num)]
        for worker in self.worker_list:
            grid_x, grid_y = self.loction_to_grid(worker.sim_actor.get_transform().location)
            worker_location_list[grid_x][grid_y]+=1
        return worker_location_list

    def get_task_location(self):
        task_location_list=[]
        for task in self.bufferPool.task_list:
            location=task.sim_actor.get_transform().location
            task_location_list.append([location.x,location.y])
        return task_location_list

    def get_task_location_grid(self):
        task_location_list =  [[0 for i in range(self.grid_num)] for j in range(self.grid_num)]
        for task in self.bufferPool.task_list:
            grid_x, grid_y = self.loction_to_grid(task.sim_actor.get_transform().location)
            task_location_list[grid_x][grid_y] += 1
        return task_location_list


    def get_road_traffic(self):
        """Obtain the number of worker congestion in each zone"""

        for zone in self.zones_list:
            zone.densities=0

        for worker in self.worker_list:
            #If the speed is less than 5km/h, it means that the workers are in congestion
            vel=worker.sim_actor.get_velocity()
            speed=3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            loc = worker.sim_actor.get_transform().location
            zone_index = self.loction_to_zone(loc)
            self.zones_list[zone_index].densities += 1

        road_traffic_list=[]

        for zone_index in self.road_network.zones:
            road_traffic_list.append(self.zones_list[zone_index].densities)

        return road_traffic_list

    def get_road_traffic_grid(self):
        road_traffic_list=[]
        for zone in self.zones_list:
            road_traffic_list.append(zone.densities)

        road_traffic_np=np.array(road_traffic_list).reshape(self.grid_num,self.grid_num)
        return road_traffic_np.tolist()

    def grade_weather(self, precipitation, wind_intensity,fog_density):
        """Grade the weather conditions into 5 levels, the higher the level,
        the worse it is, and the greater the impact on the division of time windows"""
        rank=0

        if precipitation>50:
            rank+=1
        if wind_intensity>50:
            rank+=1
        if fog_density>50:
            rank+=1

        return rank

    def get_weather(self):
        """Obtain the current  precipitation, wind speed and fog_density"""

        weather=self.world.get_weather()

        precipitation = weather.precipitation
        wind_intensity = weather.wind_intensity
        fog_density = weather.fog_density

        self.precipitation_deque.append(precipitation)
        self.wind_intensity_deque.append(wind_intensity)
        self.fog_density_deque.append(fog_density)

        rank=self.grade_weather(precipitation,wind_intensity,fog_density)
        rank_list=[[rank for i in range(self.grid_num)] for j in range(self.grid_num)]

        return rank_list

    def get_future_task(self):

        self.history_tasks_deque.append(self.cumulative_task_num)
        self.task_num_list.append(self.cumulative_task_num)
        self.cumulative_task_num=0
        task_num_list=list(self.history_tasks_deque)
        future_task_num_list=[]
        if len(task_num_list)>=20:
            # Use ARIMA algorithm to predict the number of tasks in the next three time slices
            # Build Model
            try:
                model = ARIMA(task_num_list, order=(1, 1, 1))
                # disp=1: output the convergence information -1:donot output
                fitted = model.fit(disp=-1)
                # Forecast
                fc, se, conf = fitted.forecast(5, alpha=0.05)

                future_task_num_list=list(int(fc))
            except Exception as e:
                print("ARIMA Error:",e)
                print("Error line:",e.__traceback__.tb_lineno)

        task_num_list=task_num_list+future_task_num_list

        return task_num_list

    def get_worker_condition(self):

        worker_condition_list=[]
        worker_condition={}
        id=0
        for worker in self.worker_list:

            worker_condition["ID"]=id
            worker_condition["Name"]=worker.index

            if worker.sim_actor!=None:
                vel = worker.sim_actor.get_velocity()
                speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                worker.speed_deque.append(speed)
            else:
                worker.speed_deque.append(0)
            worker_condition["Speed"] =list(worker.speed_deque)

            worker_condition["TaskNumber"] = worker.complete_task_num

            if worker.current_task!=None:
                worker_condition["Deadline"] = worker.current_task.deadline-self.world_time
            else:
                worker_condition["Deadline"] = 1

            if worker.sim_actor!=None:
                if len(worker.collision_sensor.get_collision_history())>0:
                    worker_condition["State"] = "#d10800"
                else:
                    worker_condition["State"]="#309901"
            else:
                worker_condition["State"] = "#b0b9b8"

            worker_condition["Revenue"]=worker.cumulative_revenue


            worker_condition_list.append(worker_condition)
            worker_condition={}
            id+=1


        return worker_condition_list
    

    def get_equilibrium_coefficient(self):
        workers_revenue_list = [[0 for i in range(self.grid_num)] for j in range(self.grid_num)]
        for worker in self.worker_list:
            grid_x, grid_y = self.loction_to_grid(worker.sim_actor.get_transform().location)
            workers_revenue_list[grid_x][grid_y] += worker.cumulative_revenue
        return workers_revenue_list

    def get_time_window_condition(self):
        '''
        Obtain time window related information (maximum time window. Minimum time window, current time window selection, current time window income, loss, number of training steps, total number of training steps)
        :return:
        '''
        time_window_condition={}

        time_window_condition["total_train_step"]=self.total_train_step
        time_window_condition["current_train_step"]=self.current_train_step

        time_window_condition["max_time_windows"]=list(self.max_time_window_deque)
        time_window_condition["min_time_windows"] = list(self.min_time_window_deque)
        time_window_condition["current_time_windows"] = list(self.current_time_window_deque)
        time_window_condition["time_window_revenues"]=list(self.time_window_revenue_deque)
        time_window_condition["loss"]=list(self.loss_deque)

        return time_window_condition


    def get_env_state(self,window_step):
        """Current environment state observed by agent """
        state=[]

        try:
            fea_remaining_time=self.get_worker_remaining_time_grid()
            state.append(fea_remaining_time)

            fea_deadline=self.get_task_deadline_grid()
            state.append(fea_deadline)

            fea_worker_space=self.get_worker_location_grid()
            state.append(fea_worker_space)

            fea_task_space=self.get_task_location_grid()
            state.append(fea_task_space)

            fea_road_traffic=self.get_road_traffic_grid()
            state.append(fea_road_traffic)

            fea_weather=self.get_weather()
            state.append(fea_weather)

          
            fea_equilibrium_coefficient=self.get_equilibrium_coefficient()
            state.append(fea_equilibrium_coefficient)

  
            fea_window=[[window_step for i in range(self.grid_num)] for j in range(self.grid_num)]
            state.append(fea_window)

            fea_time_slice=[[self.world_time for i in range(self.grid_num)] for j in range(self.grid_num)]
            state.append(fea_time_slice)

            #Stretch State Shape
            state=np.array(state).reshape(self.state_space).tolist()
        except Exception as e:
            print("Get State Error:", e)
            print("Error line:", e.__traceback__.tb_lineno)

        return state

    def work_thread(self,window_step,window_size):
        '''
        Work thread, the core of the entire environment, is responsible for connecting with the simulator
        :param window_step: time window step
        :param window_size: the time window size that the action action is converted to
        :return:
        '''

        # refresh buffer
        self.update_buffer_pool()

        reward=0
        reward=self.greedy_match()
        reward=random.randint(5,30)

        self.max_time_window_deque.append(self.max_window)
        self.min_time_window_deque.append(self.min_window)
        self.current_time_window_deque.append(window_size)
        self.time_window_revenue_deque.append(reward)



        next_state=self.get_env_state(window_step+1)

        return reward,next_state

    def build_topology(self):
        """
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects with the following attributes

        - entry (carla.Waypoint): waypoint of entry point of road segment
        - entryxyz (tuple): (x,y,z) of entry point of road segment
        - exit (carla.Waypoint): waypoint of exit point of road segment
        - exitxyz (tuple): (x,y,z) of exit point of road segment
        - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        """
        self.topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in self.map.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
            self.topology.append(seg_dict)

    def build_graph(self):
        """
        This function builds a networkx graph representation of topology, creating several class attributes:
        - graph (networkx.DiGraph): networkx graph representing the world map, with:
            Node properties:
                vertex: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """

        self.graph = nx.DiGraph()
        self._id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self._road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segment in self.topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self.graph.add_node(new_id, vertex=vertex)
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            # Adding edge with attributes
            self.graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW)

    def road_network_parser(self):
        """ Analyzing Edge Sets and Node Sets of Road Network Graph"""
        nodes_list=[]
        edges_list=[]
        graph=self.road_network.graph

        #Build node set
        nodes_dir = graph._node
        nodes_id = list(nodes_dir.keys())
        for i in range(len(nodes_id)):
            node=list(nodes_dir[i]['vertex'][:2])
            nodes_list.append(node)

        pos = dict(zip(nodes_id, nodes_list))

        #Build edge set
        edges=graph.edges
        for edge in edges:
            n_1=pos.get(edge[0])
            n_2=pos.get(edge[1])
            edges_list.append([n_1,n_2])

        self.road_network.nodes=nodes_list
        self.road_network.edges=edges_list

    def construct_road_zones(self):
        """Construct the road network zones"""
        self.zones_list=[]
        #Divide the road network into grid_num * grid_num grids
        self.avg_lon=(self.max_lon-self.min_lon)/self.grid_num
        self.avg_lat=(self.max_lat-self.min_lat)/self.grid_num

        for i in range(self.grid_num):
            for j in range(self.grid_num):
                zone=Zone(i*self.grid_num+j)
                if i!=self.grid_num-1:
                    zone.s_lon=self.min_lon+(i-1)*self.avg_lon
                    zone.e_lon=self.min_lon+i*self.avg_lon
                else:
                    zone.s_lon=self.min_lon+(i-1)*self.avg_lon
                    zone.e_lon=self.max_lon
                if j!=self.grid_num-1:
                    zone.s_lat = self.min_lat + (j - 1) * self.avg_lat
                    zone.e_lat = self.min_lat + j * self.avg_lat
                else:
                    zone.s_lat = self.min_lat + (j - 1) * self.avg_lat
                    zone.e_lat = self.max_lat
                self.zones_list.append(zone)

        #Map the edges of the road network to the zone
        for i in range(len(self.road_network.edges)):
            edge=self.road_network.edges[i]
            edge_lon=edge[0][0]
            edge_lat=edge[0][1]
            zone_x=int((edge_lon-self.min_lon)/self.avg_lon)
            zone_y=int((edge_lat-self.min_lat)/self.avg_lat)

            zone_index=zone_x*self.grid_num+zone_y
            if zone_index>=len(self.zones_list):
                zone_index=len(self.zones_list)-1

            self.road_network.zones.append(zone_index)


    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.worker_list[0].sim_actor.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception as e:
            print("Modify Error:",e)
            print("Error line:", e.__traceback__.tb_lineno)


    def tick(self, clock, worker):
        """Method for every tick"""
        worker.hud.worker_tick(self, worker, clock)

    def render(self, display, worker):
        """Render world"""
        worker.camera_manager.render(display)
        worker.hud.render(display)
        self.hci.render(display)

    def destroy_sensors(self,worker):
        """Destroy sensors"""
        worker.camera_manager.sensor.destroy()
        worker.camera_manager.sensor = None
        worker.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        try:
            if self.player is not None:
                self.player.destroy()

            for worker in self.worker_list:
                actors = [
                    worker.camera_manager.sensor,
                    worker.collision_sensor.sensor,
                    worker.lane_invasion_sensor.sensor,
                    worker.gnss_sensor.sensor
                    ]
                for actor in actors:
                    if actor is not None:
                        actor.destroy()

                if worker.sim_actor is not None:
                        worker.sim_actor.destroy()
                        worker.sim_actor=None
                        worker.sim_agent=None

            for task in self.bufferPool.task_list:
                if task.sim_actor is not None:
                    task.sim_actor.destroy()
        except Exception as e:
            print("Destroy World Error:", e)
            print("Error line:", e.__traceback__.tb_lineno)



class KeyboardControl(object):
    """ KeyboardControl"""

    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

class HUD(object):
    """Class for Head Up Display ( HUD ) text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def worker_tick(self, world, worker, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = worker.sim_actor.get_transform()
        vel = worker.sim_actor.get_velocity()
        control = worker.sim_actor.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = worker.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Worker: % 20s' % worker.index,
            'Vehicle: % 20s' % get_actor_display_name(worker.sim_actor, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (worker.gnss_sensor.lat, worker.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != worker.sim_actor.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_name=vehicle.attributes['role_name']
            # vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_name))

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""

        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        try:
            self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        except:
            pass

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


class TextBox:
    def __init__(self, width, height, x, y, font):
        """
        Class for TextBox
        :param width: text box width
        :param height: text box height
        :param x: text box coordinates
        :param y: text box coordinates
        """
        self.width = width
        self.height = height
        self.x = x+10
        self.y = 30
        self.text = "0" 

        self.__surface = pygame.Surface((self.width, self.height))
        if font is None:
            self.font = pygame.font.Font(None, 32)  
        else:
            self.font = font

    def draw(self, dest_surf):
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        dest_surf.blit(self.__surface, (self.x, self.y))
        dest_surf.blit(text_surf, (self.x, self.y + (self.height - text_surf.get_height())),
                       (0, 0, self.width, self.height))

    def key_down(self, event):
        unicode = event.unicode
        key = event.key

        if key == 8:
            self.text = self.text[:-1]
            return

        if key == 301:
            return

        if unicode != "":
            char = unicode
        else:
            char = chr(key)

        self.text += char


class Button(object):
    """Class of button"""
    def __init__(self, text, color, display_width, display_height,surface_span):
        font_addr = pygame.font.get_default_font()
        font = pygame.font.Font(font_addr, 20)
        self.surface = font.render(text, True, color)
        self.display_width = display_width
        self.display_height = display_height
        self.surface_span=surface_span
        self.WIDTH = self.surface.get_width()
        self.HEIGHT = self.surface.get_height()
        self.x= self.display_width+self.surface_span // 2 - self.WIDTH // 2
        # self.y= self.display_height // 8-self.HEIGHT // 2
        self.y = 100
        # self.x= self.display_width // 2 - self.WIDTH // 2
        # self.y= self.display_height // 2 - self.HEIGHT // 2


    def draw(self, screen):
        screen.blit(self.surface, (self.x, self.y))

    def check_click(self, position):
        x_match = position[0] > self.x and position[0] < self.x + self.WIDTH
        y_match = position[1] > self.y and position[1] < self.y + self.HEIGHT

        if x_match and y_match:
          return True
        else :
          return False


class DataQThread(QThread):
    """ Human computer interaction screen data update"""

    #Signal transmission between multiple threads
    # sinOut = pyqtSignal(dict)

    def __init__(self,world,human_machine_screen):
        super(DataQThread, self).__init__()

        self.world=world
        self.screen=human_machine_screen

    def number_to_color(self,num_list):
        """Converts the number to a green to red gradient"""
        color_list=[]
        colors={"green":"#057748","orange":"#ffc773","red":"#ff3300"}
        max_value=max(num_list)
        #Congestion should be judged based on speed and quantity
        value_1=max_value/5
        value_2=(max_value/5)*2
        # value_1 = 5
        # value_2 = 10

        for num in num_list:
            if num<=value_1:
                color_list.append(colors["green"])
            elif num>value_1 and num<=value_2:
                color_list.append(colors["orange"])
            elif num>value_2:
                color_list.append(colors["red"])
        return color_list

    def run(self):
        while not self.world.sim_done:

            #Update human-computer interaction screen data once every 10 seconds
            time.sleep(10)
            try:
                data_list = {}
                # 1. obtain road traffic data
                road_densities=self.world.get_road_traffic()

                road_color=self.number_to_color(road_densities)
                edge_colors=[]
                for zone_index in self.world.road_network.zones:
                    edge_colors.append(road_color[zone_index])
                data_list["edge_colors"]=edge_colors

                # 2. obtain weather data
                self.world.get_weather()
                weather_data=[]
                weather_data.append(list(self.world.precipitation_deque))
                weather_data.append(list(self.world.wind_intensity_deque))
                weather_data.append(list(self.world.fog_density_deque))
                data_list["weather"]=weather_data


                # 3.obtain spatiotemporal distribution data
                worker_remaining_time_list,worker_total_time_list=self.world.get_worker_remaining_time()
                worker_location_list=self.world.get_worker_location()
                task_deadline_list=self.world.get_task_deadline()
                task_location_list=self.world.get_task_location()
                distribution_data=[]
                distribution_data.append(worker_remaining_time_list)
                distribution_data.append(worker_total_time_list)
                distribution_data.append(worker_location_list)
                distribution_data.append(task_deadline_list)
                distribution_data.append(task_location_list)
                data_list["distribution_data"]=distribution_data

                # 4.obtain future task number data
                task_num=self.world.get_future_task()
                data_list["task_num"]=task_num

                # 5.obtain worker cumulative revenue data
                worker_condition_list=self.world.get_worker_condition()
                data_list["worker_condition"]=worker_condition_list

                # 6. obtain time window range data

                time_window_condition = world.get_time_window_condition()
                data_list["time_window_condition"] = time_window_condition

         
                #update human machine screen
                self.screen.screen_data_update(data_list)
                # self.sinOut.emit(data_list)
            except Exception as e:
                print("Human-machine data update error:",e)
                print("Error line:", e.__traceback__.tb_lineno)


class HumanWindow(QtWidgets.QWidget):

    def __init__(self,world):  
        super().__init__()  

        self.world = world
        self.road_network = self.world.road_network

        self.setGeometry(300, 300, 1800, 1000)
        self.setWindowTitle('Human-machine Interaction Screen')
        self.setWindowIcon(QIcon('images/icons/h_icon1.png'))
        self.show()
        self.layout_grid()


    def layout_grid(self):
        '''
        
        :return:
        '''

        grid = QtWidgets.QGridLayout()

        # 1. the layout of road traffic
        self.figure_1 = plt.figure()
        self.figure_1.suptitle('Road Traffic')
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.ax_1 = self.figure_1.add_axes([0.1, 0.1, 0.8, 0.8])
        grid.addWidget(self.canvas_1, 0, 0)

        # 2. the layout of weather
        self.figure_2,axes = plt.subplots(nrows=3,ncols=1,sharex=True)
        self.figure_2.suptitle('Weather')
        self.canvas_2 = FigureCanvas(self.figure_2)
        self.ax_2_0 = axes[0]
        self.ax_2_1 = axes[1]
        self.ax_2_2 = axes[2]

        grid.addWidget(self.canvas_2, 0, 1)

        # 3. the layout of spatiotemporal distribution
        self.figure_3 = plt.figure()
        self.figure_3.suptitle('Spatiotemporal Distribution')
        self.canvas_3 = FigureCanvas(self.figure_3)
        self.ax_3 = self.figure_3.add_axes([0.1, 0.1, 0.8, 0.8])
        grid.addWidget(self.canvas_3, 0, 2)

        # 4. the layout of future task number
        self.figure_4 = plt.figure()
        self.figure_4.suptitle('Future Task Number')
        self.canvas_4 = FigureCanvas(self.figure_4)
        self.ax_4 = self.figure_4.add_axes([0.1, 0.1, 0.8, 0.8])
        grid.addWidget(self.canvas_4, 1, 0)

        # 5. the layout of worker cumulative revenue
        self.figure_5 = plt.figure()
        self.figure_5.suptitle('Worker Cumulative Revenue')
        self.canvas_5 = FigureCanvas(self.figure_5)
        self.ax_5 = self.figure_5.add_axes([0.1, 0.1, 0.8, 0.8])
        grid.addWidget(self.canvas_5, 1, 1)

        # 6. the layout of time window range adjustment
        self.canvas_6=QtWidgets.QWidget()

        self.layout_6 = QtWidgets.QFormLayout()
        self.layout_6.setSpacing(20)

        hint_label = QtWidgets.QLabel("Slide the following slider to adjust time window range")
        hint_label.setStyleSheet('font-size:20px;')
        # hint_label.setFont(QFont("Roman times", 15, QFont.Bold))
        none_label = QtWidgets.QLabel("")
        # line_label = QtWidgets.QLabel("-----------------------------------------")
        line_label = QtWidgets.QLabel("============================================================")
        self.layout_6.addRow(none_label)
        self.layout_6.addRow(hint_label)
        self.layout_6.addRow(line_label)
        # self.layout_6.addRow(none_label)

        self.layout_6_0 = QtWidgets.QGridLayout()
        learn_label = QtWidgets.QLabel("Learn step")
        learn_label.setStyleSheet('font-size:20px;')
        self.layout_6_0.addWidget(learn_label, 1, 1)

        total_learn_step = self.world._args.simTimeSlice
        current_learn_step = self.world.world_time

        self.learn_step_value = QtWidgets.QLabel(str(current_learn_step) + "/" + str(total_learn_step))
        self.learn_step_value.setStyleSheet('font-size:20px;')
        self.layout_6_0.addWidget(self.learn_step_value, 1, 2)

        self.pgb = QtWidgets.QProgressBar()
        self.pgb.move(50, 50)
        self.pgb.resize(250, 15)

        self.pgb.setMinimum(0)
        self.pgb.setMaximum(total_learn_step)
        self.pgb.setValue(current_learn_step)
        self.pgb.setStyleSheet('font-size:20px;')
        self.layout_6_0.addWidget(self.pgb,1,3)

        self.layout_6.addRow(self.layout_6_0)

        #Minimum value slider for time window
        self.layout_6_1=QtWidgets.QGridLayout()
        min_label=QtWidgets.QLabel("Min window")
        min_label.setStyleSheet('font-size:20px;')
        self.layout_6_1.addWidget(min_label,1,1)

        self.min_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.min_slider.setMinimum(3)
        self.min_slider.setMaximum(15)
        self.min_slider.setSingleStep(1)
        self.min_slider.setValue(self.world.min_window)
        self.min_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.min_slider.setTickInterval(5)
        self.layout_6_1.addWidget(self.min_slider, 1, 2)

        self.min_slider_label=QtWidgets.QLabel(str(self.world.min_window))
        self.min_slider_label.setFont(QFont("Roman times", 15, QFont.Bold))
        self.min_slider.valueChanged.connect(lambda v: self.min_slider_label.setText(str(v)))
        self.layout_6_1.addWidget(self.min_slider_label, 1, 3)

        self.layout_6.addRow(self.layout_6_1)

        #Maximum value slider of time window
        self.layout_6_2 = QtWidgets.QGridLayout()
        max_label=QtWidgets.QLabel("Max window")
        max_label.setStyleSheet('font-size:20px;')
        self.layout_6_2.addWidget(max_label,1,1)

        self.max_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.max_slider.setMinimum(15)
        self.max_slider.setMaximum(33)
        self.max_slider.setSingleStep(1)
        self.max_slider.setValue(self.world.max_window)
        self.max_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.max_slider.setTickInterval(5)
        self.layout_6_2.addWidget(self.max_slider, 1, 2)

        self.max_slider_label=QtWidgets.QLabel(str(self.world.max_window))
        self.max_slider_label.setFont(QFont("Roman times", 15, QFont.Bold))
        self.max_slider.valueChanged.connect(lambda v: self.max_slider_label.setText(str(v)))
        self.layout_6_2.addWidget(self.max_slider_label, 1, 3)

        self.layout_6.addRow(self.layout_6_2)

        none_label_1 = QtWidgets.QLabel("")
        # line_label_1 = QtWidgets.QLabel("-----------------------------------------")
        # line_label_1 = QtWidgets.QLabel("======================================")
        self.layout_6.addRow(none_label_1)
        # self.layout_6.addRow(line_label_1)

        #Reset and adjust button
        self.layout_6_3 = QtWidgets.QGridLayout()
        self.reset_button=QtWidgets.QPushButton("Reset")
        self.reset_button.setStyleSheet('font-size:20px;')
        self.reset_button.clicked.connect(lambda :self.adjust_time_window(0))
        self.layout_6_3.addWidget(self.reset_button,1,1)

        self.adjust_button = QtWidgets.QPushButton("Adjust")
        self.adjust_button.setStyleSheet('font-size:20px;')
        self.adjust_button.clicked.connect(lambda: self.adjust_time_window(1))
        self.layout_6_3.addWidget(self.adjust_button, 1, 3)

        self.layout_6.addRow(self.layout_6_3)

        self.canvas_6.setLayout(self.layout_6)

        grid.addWidget(self.canvas_6, 1, 2)
        self.setLayout(grid)

    def adjust_time_window(self,button_action):
        '''
        Button response function - adjust window size

        :param button_action: 0:reset 1:confirm
        :return:
        '''
        if button_action==0:
            self.min_slider.setValue(self.world.min_window)
            self.max_slider.setValue(self.world.max_window)
        elif button_action==1:
            self.world.min_window=self.min_slider.value()
            self.world.max_window = self.max_slider.value()
            print("Time window range adjustment succeeded: [%s:%s]"
                  % (self.world.min_window,self.world.max_window))


    def screen_data_update(self,data_list):
        """Set the content of human-computer interaction screen
        """
        try:

            # 1. draw the figure of road traffic
            self.ax_1.clear()
            # self.ax_1.set_title("Road Traffic")
            edge_colors=data_list["edge_colors"]

            # draw road network
            for i in range(len(self.road_network.edges)):
                edge=self.road_network.edges[i]
                # draw edge
                self.ax_1.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=edge_colors[i], linewidth=0.8)
                # draw node
                self.ax_1.scatter(edge[0][0], edge[0][1], color="black", marker="o", s=8, alpha=0.4)
                self.ax_1.scatter(edge[1][0], edge[1][1], color="black", marker="o", s=8, alpha=0.4)

            self.canvas_1.draw()

            # 2. draw the figure of weather
            self.ax_2_0.clear()
            self.ax_2_1.clear()
            self.ax_2_2.clear()

            weather=data_list["weather"]
            precipitation_list=weather[0]
            wind_intensity_list=weather[1]
            fog_density_list=weather[2]

            self.ax_2_0.plot(precipitation_list,'.--',color='#1f77b4',label='Precipitation')
            self.ax_2_0.legend()
            self.ax_2_1.plot(wind_intensity_list, '.--', color='#ff7f0e', label='Wind Intensity')
            self.ax_2_1.legend()
            self.ax_2_2.plot(fog_density_list, '.--', color='g', label='Fog Density')
            self.ax_2_2.legend()

            self.canvas_2.draw()

            # 3. draw the figure of spatiotemporal distribution
            self.ax_3.clear()

            distribution_data=data_list["distribution_data"]
            worker_remaining_time_list = distribution_data[0]
            worker_total_time_list = distribution_data[1]
            worker_location_list = distribution_data[2]
            task_deadline_list = distribution_data[3]
            task_location_list = distribution_data[4]

            # draw road network
            for i in range(len(self.road_network.edges)):
                edge = self.road_network.edges[i]
                # draw edge
                self.ax_3.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color="black", linewidth=0.8, alpha=0.1)
                # draw node
                self.ax_3.scatter(edge[0][0], edge[0][1], color="black", marker="o", s=8, alpha=0.1)
                self.ax_3.scatter(edge[1][0], edge[1][1], color="black", marker="o", s=8, alpha=0.1)

            #draw worker location and color
            for i in range(len(worker_location_list)):
                worker_transparency=1-(worker_remaining_time_list[i]/worker_total_time_list[i])
                if worker_remaining_time_list[i]==0:
                    self.ax_3.scatter(worker_location_list[i][0], worker_location_list[i][1], color="green",
                                      marker="s", s=40, alpha=worker_transparency)
                else:
                    self.ax_3.scatter(worker_location_list[i][0], worker_location_list[i][1], color="#ff7f0e",
                                      marker="s", s=40, alpha=worker_transparency)

            #draw task location and color
            for i in range(len(task_location_list)):
                task_transparency=self.world.normalization(task_deadline_list[i],self.world.waiting_time,0)
                self.ax_3.scatter(task_location_list[i][0], task_location_list[i][1], color="#1f77b4",
                                      marker="^", s=40, alpha=task_transparency)
            #draw legend
            self.ax_3.scatter(-20, 30, color="green", marker="s", s=40, alpha=1, label="Worker")
            self.ax_3.scatter(0, 10, color="#1f77b4", marker="^", s=40, alpha=1, label="Task")
            self.ax_3.legend()
            self.canvas_3.draw()

            # 4. draw the figure of future task number
            self.ax_4.clear()

            task_num=data_list["task_num"]
            history_task_num_list=task_num[:20]
            x1 = [i for i in range(len(history_task_num_list))]

            future_task_num_list=task_num[20:]
            x2 = [i + len(x1) for i in range(len(future_task_num_list))]

            self.ax_4.plot(x1,history_task_num_list,  linestyle = '-', color='#1f77b4', label='History Number')
            self.ax_4.plot(x2,future_task_num_list, linestyle = '-.', color='#ff7f0e', label='Future Number')
            self.ax_4.legend()
            self.canvas_4.draw()

            # 5.draw the figure of worker cumulative revenue
            self.ax_5.clear()

            worker_condition_list= data_list["worker_condition"]
            self.ax_5.bar(x=range(len(worker_condition_list)), height=worker_condition_list,
                    alpha=0.5,  
                    width=0.5,  
                    color='#ffb310',  
                    edgecolor='#990033',  
                    linewidth=2,  
                    label='Revenue'

                    )
            self.ax_5.legend()

            # self.ax_5.bar(worker_condition_list,color='#ff7f0e')
            self.canvas_5.draw()

            time_window_range= data_list["time_window_range"]
            total_learn_step = self.world._args.simTimeSlice
            current_learn_step = self.world.world_time
            self.pgb.setValue(current_learn_step)
            self.learn_step_value.setText(str(current_learn_step) + "/" + str(total_learn_step))
            self.min_slider.setValue(time_window_range[0])
            self.max_slider.setValue(time_window_range[1])

        except Exception as e:
            print("Human-machine data draw error:", e)
            print("Error line:", e.__traceback__.tb_lineno)

def enable_HCIS():
    '''
    Start the human-computer interaction system
    :return:
    '''
    app.run_server(debug=True, port=8050,use_reloader=False)

def number_to_color(num_list):
    """Converts the number to a green to red gradient"""
    color_list=[]
    colors = {"green": "#057748", "orange": "#ffa631", "red": "#ff3300"}
    # max_value=max(num_list)
    # value_1=max_value/5
    # value_2=(max_value/5)*2
    value_1 = 1
    value_2 = 2

    for num in num_list:
        if num<=value_1:
            color_list.append(colors["green"])
        elif num>value_1 and num<=value_2:
            color_list.append(colors["orange"])
        elif num>value_2:
            color_list.append(colors["red"])
    return color_list

def update_globel_data(world):
    '''
    Update the global data required by the human-computer interaction system
    :param world: world object
    :return:
    '''

    while not world.sim_done:

        # Update human-computer interaction screen data once every 10 seconds
        time.sleep(10)
        try:
            global_data = gl.get_value("GLOBAL_DATA")
            if global_data[0]!="None":
                world.min_window=global_data[0]["new_min_time_window"]
                world.max_window=global_data[0]["new_max_time_window"]

            data_list = {}
            # 1. obtain road traffic data
      
            road_densities = world.get_road_traffic()

            road_color = number_to_color(road_densities)
            edge_colors = []
            for zone_index in world.road_network.zones:
                edge_colors.append(road_color[zone_index])
            data_list["edge_colors"] = edge_colors
            data_list["road_network_edges"]=world.road_network.edges

            # 2. obtain weather data
            world.get_weather()
            weather_data = []
            weather_data.append(list(world.precipitation_deque))
            weather_data.append(list(world.wind_intensity_deque))
            weather_data.append(list(world.fog_density_deque))
            data_list["weather"] = weather_data

            # 3.obtain spatiotemporal distribution data
            worker_remaining_time_list, worker_total_time_list = world.get_worker_remaining_time()
            worker_location_list = world.get_worker_location()
            task_deadline_list = world.get_task_deadline()
            task_location_list = world.get_task_location()
            distribution_data = []
            distribution_data.append(worker_remaining_time_list)
            distribution_data.append(worker_total_time_list)
            distribution_data.append(worker_location_list)
            distribution_data.append(task_deadline_list)
            distribution_data.append(task_location_list)
            task_transparency_list=[]
            for i in range(len(task_location_list)):
                task_transparency = world.normalization(task_deadline_list[i], world.waiting_time, 0)
                task_transparency_list.append(task_transparency)
            distribution_data.append(task_transparency_list)
            data_list["distribution_data"] = distribution_data

            # 4.obtain future task number data
            task_num = world.get_future_task()
            data_list["task_num"] = task_num

            # 5.obtain worker cumulative revenue data
            worker_condition_list = world.get_worker_condition()
            data_list["worker_condition"] = worker_condition_list

            # 6. obtain time window range data

            time_window_condition = world.get_time_window_condition()
            data_list["time_window_condition"] = time_window_condition

            #7. obtain current time winow
            data_list["new_max_time_window"] = world.max_window
            data_list["new_min_time_window"] = world.min_window

            gl.set_value("GLOBAL_DATA", [data_list])

        except Exception as e:
            print("Human-machine data update error:", e)
            print("Error line:", e.__traceback__.tb_lineno)


class HCI(object):
    """ Class for human-computer interaction"""

    def __init__(self, width, height, world):
        """Constructor method"""
        self.world=world
        self.dim=(width,height)
        font_addr = pygame.font.get_default_font()
        self.font = pygame.font.Font(font_addr, 15)
        self.surface_span=220
        self.surface=pygame.Surface((self.dim[0]-self.surface_span,self.dim[1]))
        self.surface_width=self.surface.get_width()
        self.surface_height=self.surface.get_height()
        self.surface.set_alpha(100)
        self.surface.fill(color='black')
        self.display_lable = self.font.render("Switch worker angle:", True, (255, 255, 255))

        self.display_text_box= TextBox(200, 30, self.surface_width, self.surface_height, self.font)

        self.display_button = Button('Display', 'red', self.surface_width+40, self.surface_height,self.surface_span)
        self.reset_button = Button('Reset', 'red', self.surface_width-40, self.surface_height, self.surface_span)
        self.current_worker_index=int(self.display_text_box.text)

     

        #Start the human-computer interaction system in multithreading
        hci_thread=threading.Thread(target=enable_HCIS)
        hci_thread.start()

        #Updating human-computer interaction system data in multithreading
        global_data_update_thread=threading.Thread(target=update_globel_data,args=[self.world])
        global_data_update_thread.start()


    def tick(self):
        """HCI method for every tick"""
        pass

    def switch_camera(self,actor_id):
        '''
        Switch to the perspective of the specified worker
        :param actor_id: worker
        :return:
        '''
        current_worker=self.world.worker_list[self.current_worker_index]

        if actor_id!=None and actor_id!="":
            actor_id = int(actor_id)
            if actor_id>=0 and actor_id<len(self.world.worker_list):
                self.world.worker=self.world.worker_list[actor_id]
                self.current_worker_index=actor_id
            else:
                current_worker.hud.notification("Please enter a worker index between 0-4", seconds=4.0)
        else:
            current_worker.hud.notification("Please enter a worker index between 0-%s" % len(self.world.worker_list),
                                        seconds=4.0)
    def reset_worker(self,actor_id):
        '''
        Respawn worker actor
        :param actor_id:
        :return:
        '''
        current_worker = self.world.worker_list[self.current_worker_index]

        if actor_id!=None and actor_id!="":
            actor_id = int(actor_id)
            if actor_id>=0 and actor_id<len(self.world.worker_list):
                worker=self.world.worker_list[actor_id]
                location=worker.sim_actor.get_transform().location
                new_point=self.world.get_near_spawn_point(location)
                worker.sim_actor.set_transform(new_point)
         
                control = worker.sim_agent.run_step()
                control.manual_gear_shift = True
               
                worker.sim_actor.apply_control(control)
                self.world.modify_vehicle_physics(worker.sim_actor)
             

            else:
                current_worker.hud.notification("Please enter a worker index between 0-4", seconds=4.0)
        else:
            current_worker.notification("Please enter a worker index between 0-%s" % len(self.world.worker_list),
                                        seconds=4.0)


    def render(self,display):
        """Render for HCI class"""

        display.blit(self.surface,(self.dim[0]-self.surface_span,0))
        display.blit(self.display_lable,(self.surface_width+10,10))

        self.display_text_box.draw(display)

        self.display_button.draw(display)

        self.reset_button.draw(display)

    def button_monitor(self):
        """Monitoring button change"""
        if self.display_button.check_click(pygame.mouse.get_pos()):
            self.display_button = Button('Display', 'red', self.surface_width-40, self.surface_height,self.surface_span)
        else:
            self.display_button = Button('Display', 'white', self.surface_width-40, self.surface_height,self.surface_span)

        if self.reset_button.check_click(pygame.mouse.get_pos()):
            self.reset_button = Button('Reset', 'red', self.surface_width+40, self.surface_height, self.surface_span)
        else:
            self.reset_button = Button('Reset', 'white', self.surface_width+40, self.surface_height, self.surface_span)

        pygame.time.delay(50)
        if pygame.mouse.get_pressed()[0]:
            if self.display_button.check_click(pygame.mouse.get_pos()):
                self.switch_camera(self.display_text_box.text)
            if self.reset_button.check_click(pygame.mouse.get_pos()):
                self.reset_worker(self.display_text_box.text)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                self.display_text_box.key_down(event)

def draw_road_network(world):
    '''
    Draw a road network
    :param world: Simulation world object
    :return:
    '''
    #save file
    save_file_name="./images/road_networks/"+world.world_name+".png"


    road_network = world.graph
    fig, ax = plt.subplots()
    nodes_dir = road_network._node
    nodes_id = list(nodes_dir.keys())
    nodes_coordinates = []
    for i in range(len(nodes_id)):
        nodes_coordinates.append(list(nodes_dir[i]['vertex'][:2]))
    pos = dict(zip(nodes_id, nodes_coordinates))

    nx.draw(road_network, pos, ax=ax, node_color='r', edge_color='b', with_labels=False, font_size=18, node_size=18)
    plt.savefig(save_file_name,dpi=400)
    plt.show()


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Worker Window")
    icon=pygame.image.load("images/icons/w_icon2.png")
    pygame.display.set_icon(icon)


    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()

        # print("world_map:",client.get_available_maps())
        # available_world_maps:
        # ['/Game/Carla/Maps/Town01', '/Game/Carla/Maps/Town01_Opt', '/Game/Carla/Maps/Town02', '/Game/Carla/Maps/Town02_Opt', '/Game/Carla/Maps/Town03',
        #  '/Game/Carla/Maps/Town03_Opt', '/Game/Carla/Maps/Town04', '/Game/Carla/Maps/Town04_Opt', '/Game/Carla/Maps/Town05', '/Game/Carla/Maps/Town05_Opt',
        #  '/Game/Carla/Maps/Town06', '/Game/Carla/Maps/Town06_Opt', '/Game/Carla/Maps/Town07', '/Game/Carla/Maps/Town07_Opt', '/Game/Carla/Maps/Town10HD',
        #  '/Game/Carla/Maps/Town10HD_Opt', '/Game/Carla/Maps/Town11/Town11']
        #default map
        # sim_world = client.get_world()
        sim_world_name='Town01'
        sim_world = client.load_world(sim_world_name, carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        sim_world.unload_map_layer(carla.MapLayer.Buildings)
        sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)


        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)
        
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)



        world = World(client.get_world(), args, sim_world_name)
        world.hci.world=world
        controller = KeyboardControl(world)
        if args.agent == "Basic":
            for worker in world.worker_list:
                worker.sim_agent=BasicAgent(worker.sim_actor)

        else:
            for worker in world.worker_list:
                worker.sim_agent =  BehaviorAgent(worker.sim_actor, behavior=args.behavior)



        for worker in world.worker_list:
            spawn_points = world.map.get_spawn_points()
            destination = random.choice(spawn_points).location
            task = world.gen_task()
            worker.sim_agent.set_destination(task.sim_actor.get_transform().location)
            worker.current_task=task


        clock = pygame.time.Clock()




        window_step = 0
        while True:
            clock.tick()
            world.world_time+=1
     
            if world.world_time%30==0:
                world.gen_task()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return


            world.hci.button_monitor()
            world.tick(clock,world.worker)
         
            pygame.display.flip()

            world.auto_gen_tasks(window_step)
            reward, next_state = world.work_thread(window_step)
            window_step+=1



       
            for worker in world.worker_list:
                if worker.sim_agent.done() and worker.current_task!=None:
                    worker.current_task=None
                    print("%s has completed the current task" % worker.index)



            for worker in world.worker_list:

                if worker.current_task!=None:
                    control = worker.sim_agent.run_step()
                    control.manual_gear_shift = False
                    worker.sim_actor.apply_control(control)

                else:
                 
                    worker.sim_actor.set_autopilot(True)
               


    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()

def sim_env():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '-map',
        help='Set up the training world map)',
        default='Town10HD_Opt',
        type=str)
    argparser.add_argument(
        '-simTimeSlice',
        help='Set the number of simulation time slices)',
        default=5000,
        type=int)
    argparser.add_argument(
        '-workerNum',
        help='Set the number of simulation worker)',
        default=5,
        type=int)
    argparser.add_argument(
        '-taskNum',
        help='Set the number of simulation task)',
        default=1000,
        type=int)
    argparser.add_argument(
        '-minWindow',
        help='Set minimum time window size',
        default=5,
        type=int)
    argparser.add_argument(
        '-maxWindow',
        help='Set maximum time window size',
        default=20,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)

    sim_env()
    print("Simulation Over!!!")
    sys.exit(app.exec_())
