import os
import sys
from sim_env import World,HUD,KeyboardControl,HumanWindow,CollisionSensor
import models.DQN.model as DQN_brain
import argparse
import logging
import random
import numpy as np
import glob
from tqdm import tqdm
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout

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
except Exception as e:
    print("e:", e)

# Add PythonAPI for release mode
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except Exception as e:
    print("e:", e)

import carla

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import vector

def RL_decision(world,state,agent,window_step):
    '''
    RL decision
    :param args: hyperparameter list
    :param time_slice: current time slice
    :param window_step: number of window steps
    :return:
    '''
    divide_flag=0

    action=agent.choose_action(torch.FloatTensor(state))

    window_size=world.action_space[action]

    #Limit the action selection range of the agent
    if window_size>=world.min_window and window_size<world.max_window:
        if window_size-window_step<3 and window_size-window_step>-3:
            divide_flag=1

    if window_size>world.max_window:
        divide_flag=1
        window_size=random.randint(world.min_window,world.max_window)

    if window_size<world.min_window:
        window_size=random.randint(world.min_window,world.max_window)

    return divide_flag,action,window_size

def Simulation(args):
    '''
    Simulator, the main interface of the entire training environment
    :param args:
    :return:
    '''

    world = None
    pygame.init()
    pygame.font.init()
    # pygame.display.set_caption("Worker Window")
    # icon = pygame.image.load("images/icons/w_icon2.png")
    # pygame.display.set_icon(icon)


    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()

        sim_world=client.load_world(args.map,carla.MapLayer.Buildings| carla.MapLayer.ParkedVehicles)
        sim_world.unload_map_layer(carla.MapLayer.Buildings)
        sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

   

        world = World(client.get_world(), args,  args.map)
        world.hci.world = world

        controller = KeyboardControl(world)

        if args.agent == "Basic":

            for worker in world.worker_list:
                worker.sim_agent = BasicAgent(worker.sim_actor)
        else:
            for worker in world.worker_list:
                worker.sim_agent = BehaviorAgent(worker.sim_actor, behavior=args.behavior)
        # for worker in world.worker_list:
        #     spawn_points = world.map.get_spawn_points()
        #     destination = random.choice(spawn_points).location
        #     # worker.sim_agent.set_destination(destination)
        #     task = world.gen_task()
        #     worker.sim_agent.set_destination(task.sim_actor.get_transform().location)
        #     worker.current_task = task

        # world.player.destroy()
        window_step=0

        clock = pygame.time.Clock()

        
        agent = DQN_brain.DQN(world.state_space,
                              world.action_space,
                              lr=0.01,
                              epsilon=0.9,
                              gamma=0.9,
                              target_replace_iter=100,
                              memory_capacity=2000,
                              batch_size=32
                              )


        models_dir="./models/DQN/models/"
        models_files=os.listdir(models_dir)
        if models_files is not None and len(models_files)!=0:
            new_models_file = models_files[-1]
            #Load the latest model
            agent.load(models_dir+new_models_file)
            print("Loading model succeeded:", models_dir+new_models_file)
        else:
            print("No available models found.")


        world.total_train_step=args.simTimeSlice
        for i in tqdm(range(args.simTimeSlice)):
            try:
                clock.tick()

                world.current_train_step=i

                world.world_time=i

                if args.sync:
                    world.world.tick()
                else:
                    world.world.wait_for_tick()

                if controller.parse_events():
                    return


                # world.hci.button_monitor()
                # world.tick(clock, world.worker)
                # world.render(display, world.worker)

                # pygame.display.flip()

                world.auto_gen_tasks(i)

                #change wheather
                if world.world_time%1==0:
                    world.change_weather()

                state=world.get_env_state(window_step)
                divide_flag,action,window_size=RL_decision(world,state,agent,window_step)

                if divide_flag:
                    reward,next_state=world.work_thread(window_step,window_size)
                    window_step=0
                    world.total_reward+=reward

                else:
                    window_step+=1
                    reward=0
                    next_state=state

                agent.store_transition(state, action, reward, next_state)

                if agent.memory_counter > 100:
                    loss=agent.learn()
                    world.loss_deque.append(loss)

                for worker in world.worker_list:
                    if worker.sim_agent.done() and worker.current_task != None:
                        worker.current_task = None
                        world.total_com_task_num+=1

                        print("%s has completed the current task" % worker.index)


                for worker in world.worker_list:

                    if worker.current_task != None:
                        if len(worker.collision_sensor.get_collision_history())>2:
                         
                            world.reset_worker(worker)
                        control = worker.sim_agent.run_step()
                        control.manual_gear_shift = False
                        worker.sim_actor.apply_control(control)
                    else:
                        if len(world.bufferPool.task_list)>0:
                            task=world.bufferPool.task_list[0]
                            worker.current_task = task
                            # Calculate revenue
                            worker.cumulative_revenue += world.get_reward(worker,task)
                            worker.complete_task_num += 1
                            worker.sim_agent.set_destination(task.location)
                            world.bufferPool.task_list.remove(task)
                        else:
                          
                            worker.sim_actor.set_autopilot(True)


                if i % 500 == 0:
                    agent.save()
            except Exception as e:
                print("Learn Error:", e)
                print("Error line:", e.__traceback__.tb_lineno)


        print("total_reward:", world.total_reward)
        print("total gen tasks:", world.total_gen_task_num)
        print("total com tasks:", world.total_com_task_num)
        world.sim_done = True

        if world is not None:
            world.destroy()


    finally:
        print("total_reward:", world.total_reward)
        print("total gen tasks:", world.total_gen_task_num)
        print("total com tasks:", world.total_com_task_num)
        #Mark the end of simulation and close the human-computer screen
        world.sim_done=True

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            world.destroy()
        # pygame.quit()


if __name__=="__main__":

    argparser = argparse.ArgumentParser(
        description='CARLA Spatial Crowdsourcing Control Client')
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
        default='vehicle.audi.*',
        help='Actor filter (default: "vehicle.a1udi.*")')
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
        default=20,
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


    try:
        app = QApplication(sys.argv)
        Simulation(args)
        sys.exit(app.exec_())


    except KeyboardInterrupt:
        print('\nSimulation Over!!!')