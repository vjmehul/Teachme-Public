'''
This demo of a TAMER algorithm implmented with HIPPO Gym has been adapted
from code provided by Calarina Muslimani of the Intelligent Robot Learning Laboratory
To use this code with the default setup simply rename this file to agent.py
'''

from turtle import pd
import gym
import time
import numpy as np
import itertools
import Q_Learning as QL
import wandb
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta

#from gym import wrappers

#import highway_env
#from pettingzoo.mpe import simple_v2

import griddly
from griddly import gd
#from Policy_Shaping import PS

#global state

#from gym.utils import play
#import wandb



#This is the code for tile coding features
basehash = hash

'''
This is a demo file to be replaced by the researcher as required.
This file is imported by trial.py and trial.py will call:
start()
step()
render()
reset()
close()
These functions are mandatory. This file contains minimum working versions
of these functions, adapt as required for individual research goals.
'''
import Policy_Shaping as PS
from Policy_Shaping_copy import get_state
import math


class LevelGenerator:

    def __init__(self, config):
        self._config = config

    def generate(self):
        raise NotImplementedError()


# rapper.build_gym_from_yaml('SokobanTutorial', 'sokoban.yaml', level=0)
class ClustersLevelGenerator(LevelGenerator):
    # BLUE_BLOCK = 'a'
    # BLUE_BOX = '1'
    # RED_BLOCK = 'b'
    # RED_BOX = '2'
    # GREEN_BLOCK = 'c'
    # GREEN_BOX = '3'
    EXIT = 'x'
    AGENT = 'A'

    WALL = 'w'
    SPIKES = 'h'

    def __init__(self, config):
        super().__init__(config)
        self._width = config.get('width', 16)
        self._height = config.get('height', 14)
        self._m_spike = config.get('m_spike', 5)
        self._m_exit = config.get('m_exit', 5)

    def _place_walls(self, map):

        # top/bottom wall

        wall_y = np.array([0, self._height - 1])
        map[:, wall_y] = ClustersLevelGenerator.WALL
        map[3, 9] = ClustersLevelGenerator.WALL

        # left/right wall
        wall_x = np.array([0, self._width - 1])
        map[wall_x, :] = ClustersLevelGenerator.WALL

        return map

    def generate(self):
        map = np.chararray((self._width, self._height), itemsize=2)
        map[:] = '.'
        # print(map)
        # Generate walls
        map = self._place_walls(map)
        #map.tofile('test.txt')
        # print(map)

        # all possible locations
        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                possible_locations.append([w, h])

        def addChar(text, char, place):
            return text[:place - 1] + char + text[place + 1:]

        level_string = """w w w w w w w w w w w w w w w w
        w w . . . . . w w w . . . . x w
        w w . w w w . w w w . w w w w w
        w w . w . w . . . . . . . w t w
        w w . w . w w w w . w w w w . w
        w . . . . . . w w w w . . . . w
        w . w w w w . w w w w . w w w w
        w . . . . w . . . . . . . . . w
        w w w w w w . w w w w . w w . w
        w . . . . . . . . . . . . . . w
        w . w w w w . w w w . w w w . w
        w . w . w w . w w w . w w w w w
        w . w . . . . . t . . . . . . w
        w w w w w w w w w w w w w w w w"""

        places = "."

        occurrences = level_string.count(places)

        indices = [i for i, a in enumerate(level_string) if a == places]

        #print(indices)

        # Place Agent
        agent_location_idx = np.random.choice(indices)
        level_string = addChar(level_string, "A", agent_location_idx)
        # print(level_string)
        # agent_location = possible_locations[agent_location_idx]
        # print(agent_location_idx)
        # map[agent_location[0], agent_location[1]] = ClustersLevelGenerator.AGENT

        # level_string = ''
        # for h in range(0, self._height):
        #     for w in range(0, self._width):
        #         level_string += map[w, h].decode().ljust(4)
        #     level_string += '\n'

        # print(type(level_string))
        return level_string




#print(reward, feedback)
def update_feedback(reward):
    # feedback = 0
    if reward == "good":
        return 0.2
    elif reward == "bad":
        return -0.2
    elif reward == "None":
        return 0


epsilon_max = 1
epsilon_min = 0.1
eps_decay = 3000

weight_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)

def get_state(state):
    if len(state) != 2:
        s = np.array(np.where(state[0] == 1)).T.flatten()
        if s != []:
            x = s[0]
            y = s[1]
            if (x, y) != (0, 0):

                return (str(x), str(y))
            else:
                return (str(1), str(1))

        else:
            return (str(1), str(1))

    else:
        return state


class Agent():
    '''
    Use this class as a convenient place to store agent state.
    '''

    def start(self, game:str):
        '''
        Starts an OpenAI gym environment.
        Caller:
            - Trial.start()
        Inputs:
            -   game (Type: str corresponding to allowable gym environments)
        Returs:
            - env (Type: OpenAI gym Environment as returned by gym.make())
            Mandatory
        '''

        self.config = {
            'width': 16,
            'height': 14
        }

        level_generator = ClustersLevelGenerator(self.config)
        self.total_reward = 0
        self.demo_steps = 100
        self.feedback_steps = 100
        self.demo = False
        self.env = gym.make("GDY-Labyrinth-v0", player_observer_type=gd.ObserverType.VECTOR, level = 2, max_steps = 1000)
        #self.env.reset(level_string=level_generator.generate())
        self.env.reset()
        self.action_space = self.env.action_space.n
        # print(action_space)
        self.observation_space = self.env.observation_space.shape
        # print(observation_space)
        self.PS = True
        if self.PS:
            np.random.seed(0)
            self.PolSh = PS.PSAgent(self.action_space, self.observation_space)
            self.Qagent = QL.QLAgent(self.action_space, self.observation_space, epsilon=0.2, mini_epsilon=0.01,
                                     decay=0.999)

        wandb.init(project='SocialAILabs',name='Griddly_test')
        #wandb variables
        self.human_feeback_bad_total = 0
        self.human_feeback_good_total = 0
        self.human_feeback_bad = 0
        self.human_feeback_good = 0
        self.human_feeback_total = 0
        
        self.game_num=1
        self.game_num_list=[]

        self.data_HF_bad_list = []
        self.data_HF_good_list = []

        self.data_HF_bad_total_list = []
        self.data_HF_good_total_list = []
        
        self.feedback_reward_cumulative = 0

        self.timestep_main=0
        self.time_step_list=[]

        self.total_test_bar = []

        self.demo_list=[]
        self.total_democount=0
        self.demo_reward_cumulative = 0

        self.dataframe= pd.DataFrame()

        self.exp_start_time = timer()
        self.game_start_time = timer()

        self.game_elapsed_time_list=[]
        return


    def step(self, human_action, human_feedback):
        '''
        Takes a game step.
        Caller:
            - Trial.take_step()
        Inputs:
            - env (Type: OpenAI gym Environment)
            - action (Type: int corresponding to action in env.action_space)
        Returns:
            - envState (Type: dict containing all information to be recorded for future use)
              change contents of dict as desired, but return must be type dict.
        '''
        self.timestep_main += 1
        if self.demo == False:
            if self.time_step == 0:
                self.state =self.first_state
                time.sleep(1.5)
                
            self.time_step += 1
            
            self.last_state = np.copy(self.state)
            #print((self.last_state))
            #Q_prob = self.Qagent.action_prob(state)
            #P_prob = self.PolSh.Agent.action_prob(self.state)
            self.cnt += 1
            weight = weight_by_frame(self.cnt)
            #print(weight)

            prob = self.Qagent.action_prob(self.last_state) + weight * np.asarray(self.PolSh.action_prob(self.last_state))
            #print(prob, sum(prob)) 
            
            prob = np.asarray(prob)
            #sum = np.sum(prob)
            #prob = prob/sum
            #prob = prob/5
            #print(prob)

            #print(prob==prob.max())
            action = np.random.choice(np.flatnonzero(prob == prob.max())) + 1
            #action_space = list(range(1, self.action_space))
            #action = np.random.choice(action_space, p=prob / sum(prob))

            #print(action)
            #prob_d =
            #a = [1,2,3,4]
            #action = np.random.choice(a,p = prob)
            #stochastic probability distribution
            feedback = update_feedback(human_feedback)
            self.feedback_reward_cumulative += feedback
            next_state, reward, done, info = self.env.step(action)
            if action != 0 and self.feedback_steps > 0:
                self.PolSh.learning(action, feedback, self.last_state, next_state)
                self.Qagent.learning(action, reward, self.last_state, next_state)

                if feedback>0:
                    self.human_feeback_good+=1
                    self.human_feeback_good_total+=1
                    self.human_feeback_total +=1
                if feedback<0:
                    self.human_feeback_bad+=1
                    self.human_feeback_bad_total+=1
                    self.human_feeback_total+=1
                if feedback != 0:
                    self.feedback_steps -= 1


            self.state = next_state
            self.total_reward += reward

        elif self.demo == True:
            if self.demo_steps == 0:
                self.demo = False
            feedback_demo = 1
            self.demo_reward_cumulative += 1
            self.last_state = np.copy(self.state)
            next_state, reward, done, info = self.env.step(human_action)
            #PSA.next_state = next_state
            if human_action != 0 and self.demo_steps > 0:
               self.PolSh.learning(human_action, feedback_demo, self.last_state, next_state)
               self.Qagent.learning(human_action, reward, self.last_state, next_state)
               #print(get_state(self.last_state), get_state(self.state))
               self.demo_steps -= 1
               self.total_democount+=1
            action = human_action
            #PSA.state = next_state
            self.state = next_state
            self.total_reward += reward
            if done == True:
                self.demo = False

        if done: 
            self.game_num_list.append(self.game_num)
            self.game_elapsed_time_list.append(timedelta(seconds=timer()-self.game_start_time).seconds)
            self.game_start_time = timer()

            self.game_num+=1
            self.human_feeback_bad=0
            self.human_feeback_good=0



        self.time_step_list.append(self.timestep_main)
        self.data_HF_bad_list.append(self.human_feeback_bad)
        self.data_HF_good_list.append(self.human_feeback_good)
        
        self.data_HF_bad_total_list.append(self.human_feeback_bad_total)
        self.data_HF_good_total_list.append(self.human_feeback_good_total)

        self.dataframe = self.dataframe.append(
            {
                "time_step": self.timestep_main,
                "Game_num": self.game_num,
                "Total_bad_feeback": self.human_feeback_bad_total,
                "Total_good_feeedback": self.human_feeback_good_total,
                "Total_Reward": self.total_reward,
                "Total_demo_steps": self.demo_steps,
                "elapsed_time":timedelta(seconds=timer()-self.exp_start_time)

            },
            ignore_index=True
        )



        if done:
            wandb.log(
                {"Human_feedback" : wandb.plot.line_series(
                        xs=self.time_step_list,
                        ys=[self.data_HF_bad_list, self.data_HF_good_list],
                        keys=["Bad Feedback", "Good Feedback"],
                        title="Human feedback",
                        xname="Time step")}
            )

            wandb.log(
                {"Human_feedback_total" : wandb.plot.line_series(
                        xs=self.time_step_list,
                        ys=[self.data_HF_bad_total_list, self.data_HF_good_total_list],
                        keys=["Total Bad Feedback", "Total Good Feedback"],
                        title="Human feedback Total",
                        xname="Time step")}
                    )


            data = [[game_num, Elapsed_time] for (game_num, Elapsed_time) in zip(self.game_num_list, self.game_elapsed_time_list)]
            table = wandb.Table(data=data, columns = ["Game_num", "Time_taken"])
            wandb.log({"Elapsed_time_per_game" : wandb.plot.bar(table, "Game_num", "Time_taken", title="Elapsed time every game")})
            

            tbl = wandb.Table(data=self.dataframe)
            wandb.log({"Time_step_data": tbl})



        wandb.log(
            {
            "Total_rewards":self.total_reward,
            "human_feeback_total":self.human_feeback_total,
            "human_feeback_good_ingame":self.human_feeback_good,
            "human_feeback_bad_ingame":self.human_feeback_bad,
            "human_feeback_bad_total":self.human_feeback_bad_total,
            "human_feeback_good_total":self.human_feeback_good_total,
            "Game_num": self.game_num,
            "Demo_count": self.total_democount,
            "elapsed_time_seconds":timedelta(seconds=timer()-self.exp_start_time).seconds,
            "elapsed_time_minutes":(timedelta(seconds=timer()-self.exp_start_time).seconds)/60,
            "Reward_from_demo":self.demo_reward_cumulative,
            "Reward_feedback_cumulative": self.feedback_reward_cumulative
            }
            )


        print(self.timestep_main, self.game_num)
        envState = {'observation': next_state, 'reward': reward, 'done': done, 'info': info, 'agentAction': action}


        return envState

    def render(self):
        '''
        Gets render from gym.
        Caller:
            - Trial.get_render()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            - return from env.render('rgb_array') (Type: npArray)
              must return the unchanged rgb_array
        '''
        return self.env.render('rgb_array', observer = "global")

    def reset(self):
        '''
        Resets the environment to start new episode.
        Caller:
            - Trial.reset()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        if self.PS:
            self.cnt = 0
            self.time_step = 0
            #level_generator = ClustersLevelGenerator(self.config)
            #self.first_state = self.env.reset(level_string=level_generator.generate())
            self.first_state = self.env.reset()
            #state = self.PolSh.first_state
        else:
            #level_generator = ClustersLevelGenerator(self.config)
            #self.env.reset(level_string=level_generator.generate())
            self.env.reset()

        
    def close(self):
        '''
        Closes the environment at the end of the trial.
        Caller:
            - Trial.close()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        self.env.close()
        

if __name__ == '__main__':
    pass