import os
import sys
from sim_env import World,HUD,KeyboardControl
import models.PPO.model as PPO_brain
import argparse
import logging
import random
import glob
from tqdm import tqdm
import torch
import numpy as np
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
except IndexError:
    pass

# Add PythonAPI for release mode
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

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

    max_action, actions, log_probability = agent.get_action(torch.FloatTensor(state))


    window_size=world.action_space[max_action]

    #Limit the action selection range of the agent
    if window_size>=world.min_window and window_size<world.max_window:
        if window_size-window_step<3 and window_size-window_step>-3:
            divide_flag=1
    if window_size > world.max_window:
        divide_flag = 1

    return divide_flag, actions, log_probability

def Simulation(args):
    '''
    Simulator, the main interface of the entire training environment
    :param args:
    :return:
    '''

    world = None
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Worker Window")
    icon = pygame.image.load("images/icons/w_icon2.png")
    pygame.display.set_icon(icon)


    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()

        sim_world=client.load_world(args.map)
        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        world = World(client.get_world(), args,  args.map)
        world.hci.world = world

        controller = KeyboardControl(world)

        if args.agent == "Basic":

            for worker in world.worker_list:
                worker.sim_agent = BasicAgent(worker.sim_actor)
        else:
            for worker in world.worker_list:
                worker.sim_agent = BehaviorAgent(worker.sim_actor, behavior=args.behavior)

        # world.player.destroy()
        window_step=0

        clock = pygame.time.Clock()

        
        agent = PPO_brain.PPO(world.state_space,
                              world.action_space,
                              lr=0.005,
                              gamma=0.95,
                              clip=0.2,
                              timesteps_per_batch=480,
                              max_timesteps_per_episode=160,
                              n_updates_per_iter=5
                              )

        models_dir = "./models/PPO/models/"
        models_files = os.listdir(models_dir)
        if models_files is not None and len(models_files) != 0:
            new_models_file = models_files[-1]
            # Load the latest model
            agent.load(models_dir + new_models_file)
            print("Loading model succeeded:", models_dir + new_models_file)
        else:
            print("No available models found.")


        batch_observations = []
        batch_actions = []
        batch_logprobabilities = []
        batch_rewards = []
        batch_rewardstogo = []
        batch_lens = []
        t_batch = 0
        ep_r = []
        ep_t = 0


        for i in tqdm(range(args.simTimeSlice)):
            try:
                clock.tick()

      
                world.world_time=i

                if args.sync:
                    world.world.tick()
                else:
                    world.world.wait_for_tick()

                if controller.parse_events():
                    return

                world.hci.button_monitor()
                world.tick(clock, world.worker)
                world.render(display, world.worker)

                pygame.display.flip()

                world.auto_gen_tasks(i)

                #  Collect data
                if ep_t < agent.max_timesteps_per_episode:
                    state = world.get_env_state(window_step)
                    batch_observations.append(state)
                    divide_flag, actions, log_probability = RL_decision(world, state, agent, window_step)

                    if divide_flag:
                        reward, next_state = world.work_thread(window_step)
                        window_step = 0
                        world.total_reward += reward

                    else:
                        reward = 0
                        window_step += 1

                    # store transitions
                    ep_r.append(reward)
                    batch_actions.append(actions)
                    batch_logprobabilities.append(log_probability)

                    ep_t+=1
                    t_batch += 1

                else:
                    batch_lens.append(ep_t + 1)
                    batch_rewards.append(ep_r)
                    ep_t=0
                    ep_r=[]

                # Model learning
                if t_batch == agent.timesteps_per_batch:
                    batch_lens.append(ep_t + 1)
                    batch_rewards.append(ep_r)
                    ep_t = 0
                    ep_r = []

                    batch_observations = torch.FloatTensor(np.array(batch_observations))
                    batch_actions = torch.FloatTensor(np.array(batch_actions))
                    batch_logprobabilities = torch.FloatTensor(
                        np.array(batch_logprobabilities))

                    # compute rewars-to-go R^t
                    batch_rewardstogo = agent.compute_rewardtogo(batch_rewards)

                    agent.learn(i,batch_observations, batch_actions, batch_logprobabilities, batch_rewards, batch_rewardstogo, batch_lens)

                    batch_observations = []
                    batch_actions = []
                    batch_logprobabilities = []
                    batch_rewards = []
                    batch_rewardstogo = []
                    batch_lens = []
                    t_batch = 0

                for worker in world.worker_list:
                    if worker.sim_agent.done() and worker.current_task != None:
                        worker.current_task = None
                        world.total_com_task_num+=1
                        print("%s has completed the current task" % worker.index)


                for worker in world.worker_list:
                    # world.modify_vehicle_physics(worker.sim_actor)
                    # worker.sim_actor.set_autopilot(True)
                    if worker.current_task != None:
                        try:
                            control = worker.sim_agent.run_step()
                            control.manual_gear_shift = False
                            worker.sim_actor.apply_control(control)
                        except:
                            pass
                   
                    else:
                        try:
                            control = worker.sim_agent.run_step()
                            control.manual_gear_shift = True
                            worker.sim_actor.apply_control(control)
                        except:
                            pass
                    

                if i % 500 == 0:
                    agent.save()
            except Exception as e:
                print("Learn Error:", e)
                print("Error line:", e.__traceback__.tb_lineno)

    finally:
        print("total_reward:", world.total_reward)
        print("total gen tasks:", world.total_gen_task_num)
        print("total com tasks:", world.total_com_task_num)
        # Mark the end of simulation and close the human-computer screen
        world.sim_done = True

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            try:
                world.destroy()
            except Exception as e:
                print("Destroy World Error:", e)
                print("Error line:", e.__traceback__.tb_lineno)

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
        default=2000,
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


    try:
        app = QApplication(sys.argv)
        Simulation(args)
        sys.exit(app.exec_())

    except KeyboardInterrupt:
        print('\nSimulation Over!!!')