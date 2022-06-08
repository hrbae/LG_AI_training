# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:02:18 2022

@author: DoublekPark
"""

import pandas as pd
import numpy as np
import math
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from resizeimage import resizeimage
import pylab

import random
import copy
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parameter = pd.read_csv("D:\\Dropbox\\1. 학교과제\\2022년\\웰스테크\\4.개발\\개발 구성도\\state_parameter.csv")
parameter2_1 = parameter[parameter["index"] == 0]
parameter2_2 = parameter[parameter["index"] == 1]

len_action = parameter.shape[0]


class manufacturing:
    
    def __init__(self, parameter):
        
        self.parameter = parameter

    def producing(self, product, order):
        
        parameter2 = self.parameter[self.parameter["index"]==product]
        parameter2 = parameter2.reset_index(drop=True)
        
        time = parameter2["time"][0]
        shape = parameter2["shape"][0]
        scale = parameter2["scale"][0]
        
        rg = np.random.gamma(shape, scale, 1)
        
        y = time * order + rg
        
        return(y)
    
    def changing(self, before_product, product):
        
        
        parameter2_1 = self.parameter[self.parameter["index"]==product]
        parameter2_1 = parameter2_1.reset_index(drop=True)    
        shape1 = parameter2_1["shape"][0]
        scale1 = parameter2_1["scale"][0]
        
        y1 = np.random.gamma(shape1, scale1, 1)
        
        parameter2_2 = self.parameter[self.parameter["index"]==product]
        parameter2_2 = parameter2_2.reset_index(drop=True)    
        shape2 = parameter2_2["shape"][0]
        scale2 = parameter2_2["scale"][0]
        
        y2 = np.random.gamma(shape2, scale2, 1)
        
        y3 = y1 + y2
        
        return(round(y3[0]))
    

    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)    

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        return self.head(x.view(x.size(0), -1))   
    
#### Environment

def Generate_Env(order_a, order_b, order_c, order_d, order_e):
    
    grid = pd.DataFrame({"A":order_a,
                         "B":order_b,
                         "C":order_c,
                         "D":order_d,
                         "E":order_e},
                         index = [0])
              
    return(grid)


parameter["stocks2"] = parameter["stocks"]

action = [ 1, 0, 1, 0, 1, 2, 3, 4, 1, 0, 3, 4, 19]
a = 0
state = copy.deepcopy(parameter) 
next_state = copy.deepcopy(state)

mf = manufacturing(parameter=parameter)
mf.producing(product = 0, order = 20)
action_len = parameter.shape[0]

count1= 20
class Modeling:
    
    def __init__(self, p_count, mf, parameter):
        
        # self.Action = [0, 1, 2, 3, 4]
        
        # img = Image.fromarray(grid_m.to_numpy(), 'RGB')
        # img = resizeimage.resize_cover(img, [5,6])
        
        # self.state_size = [img.size[0],img.size[1],3]
        # self.action_size = len(self.Action)
        
        # self.n_actions = len(self.Action)
        self.p_count = p_count
        self.parameter = parameter
        
        self.mf = mf
        
    def get_state(self,
                  state,
                  action,
                  count1):
        
        a = 0
        penalty1_list = []
        penalty2_list = []
        timelist = []
        
        next_state = copy.deepcopy(state)
        inventory_df = np.zeros((len(action), state.shape[0]))
        inventory_df = pd.DataFrame(inventory_df)

        while True:
            
            # print("========================")
            # print("Action: " + str(action[a]))
            # print("Action: " + str(a))
            penalty1 = 0
            penalty2 = 0
            time = 0
            
            inventory_df.loc[a,:] = next_state.loc[:,"stocks2"]
            
            if a == 0:
                
                before_action = 100
                
            else:
                
                before_action = action[a-1]            
                
            if before_action != action[a]:
                
                penalty1 = 0
            
            else:
                
                penalty1 = -1
                
            # next_state = copy.deepcopy(state)
            before_product = next_state.loc[action[a], "index2"]
            
            if before_product == -1:
            
                next_state.loc[action[a],"stocks2"] += count1
                
                # print("-1")
                # print(next_state.loc[:,"stocks2"])
                
            else:
                
                if next_state.loc[next_state["index"]==before_product, "stocks2"].values[0] >= count1:
                    
                    next_state.loc[next_state["index"]==before_product,"stocks2"] -= count1
                    next_state.loc[action[a],"stocks2"] += count1
                    
                    # print("not -1")
                    # print(next_state.loc[:,"stocks2"])
                    
                else:
                    
                    # print("Penalty 2")
                    penalty2 = -10
                
            
            
            p_time = self.mf.producing(product=action[a], order=count1)
            c_time = self.mf.changing(before_product=before_action,
                                      product=action[a])  
            
            time = p_time + c_time
            
            penalty1_list.append(penalty1)
            penalty2_list.append(penalty2)
            timelist.append(round(time[0],2))
            
            # state = copy.deepcopy(next_state)
            
            a += 1

            if penalty2 == -10:
                
                break
            
            if a == len(action):
                
                break
            
        # print(next_state)

        return(timelist, penalty1_list, penalty2_list, next_state, inventory_df.loc[:a-1,:])
        

    def select_action(self, seq_len):
        
        action_len = self.parameter.shape[0]
        action_seq = []
        
        for i in range(seq_len):
            
            action = random.randint(0, action_len-1)
            action_seq.append(action)
            
        return(action_seq)
        

simulation = Modeling(p_count=100, mf=mf, parameter=parameter)
order = [100, 100, 100, 100, 100]

episodes = 1000

df_total = pd.DataFrame()

for e in range(episodes):
    
    action = simulation.select_action(seq_len = 100)
    
    state = copy.deepcopy(parameter)
    result = simulation.get_state(state=state, action=action, count1=20)
    timelist = result[0]
    penalty1 = result[1]
    penalty2 = result[2]
    final_state = result[3]
    inventory = result[4]
    
    df = pd.DataFrame({
        "simulation no": e,
        "seq":list(range(len(timelist))),
        "action": action[:len(timelist)],
        "time":timelist,
        "penalty1":penalty1,
        "penalty2":penalty2})
    
    df2 = pd.concat([df, inventory], axis=1)
    
    df_total = df_total.append(df2)
    
        
df_total = df_total.reset_index(drop=True)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        