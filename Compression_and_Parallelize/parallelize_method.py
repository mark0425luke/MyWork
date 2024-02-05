import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
# from models.Pruneable import Pruneable
# from utils.constants import PROD_SMALL_POOL, SMALL_POOL
from torch.autograd.function import InplaceFunction, Function
import torch.nn.functional as F
import fileinput
import itertools
import math
import random
from prettytable import PrettyTable
import logging
import argparse
import pickle
# from scipy.optimize import differential_evolution
import pulp
import Config
from Compression_and_Parallelize import OU_Table
from Compression_and_Parallelize import data_preprocess
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
logger = logging.getLogger()

def parallelize_OU_to_Macro(cycle_for_each_OU, num_Macro, num_OU_per_Macro):
    # Define the problem
    prob = pulp.LpProblem("LoadBalancing", pulp.LpMinimize)

    # Constants
    groups = ["G" + str(i+1) for i in range(len(cycle_for_each_OU))]
    machines = ["M" + str(i+1) for i in range(num_Macro)]
    group_loads = {groups[i]: cycle_for_each_OU[i] for i in range(len(cycle_for_each_OU))}

    # Decision variables
    # x_ij = 1 if group i is assigned to machine j, 0 otherwise
    x = pulp.LpVariable.dicts("x", (groups, machines), cat="Binary")

    # f_ij represents the fraction of group i that is assigned to machine j
    f = pulp.LpVariable.dicts("f", (groups, machines), lowBound=0, upBound=1, cat="Continuous")

    # Objective function: Maximize the balance across the machines
    M = pulp.LpVariable("M", lowBound=0)  # This will represent the maximum load on any machine
    prob += M, "Objective"

    # Load balancing constraints
    for machine in machines:
        prob += pulp.lpSum(group_loads[group] * f[group][machine] for group in groups) <= M, f"BalancingConstraint_{machine}"

    # Each group must be assigned to at least one machine
    for group in groups:
        prob += pulp.lpSum(x[group][machine] for machine in machines) >= 1, f"AssignmentConstraint_{group}"

    # Fraction of group load assignment constraint
    for group in groups:
        prob += pulp.lpSum(f[group][machine] for machine in machines) == 1, f"FractionLoadAssignment_{group}"

    # Constraint to ensure f_ij is only non-zero when x_ij = 1
    for group in groups:
        for machine in machines:
            prob += f[group][machine] <= x[group][machine], f"InteractionConstraint_{group}_{machine}"

    # num_OU_per_Macro constraint: Each machine can handle at most num_OU_per_Macro groups
    for machine in machines:
        prob += pulp.lpSum(x[group][machine] for group in groups) <= num_OU_per_Macro, f"OU_Constraint_{machine}"

    # Solve the problem
    # prob.solve()
    logging.info("start prob.solve")
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=10)) # 可以設定最多跑多久, 原本適用 maxSeconds，但是maxSeconds 好像官網寫棄用了
    # prob.solve(pulp.PULP_CBC_CMD(maxSeconds=900)) # 可以設定最多跑多久, 原本適用 maxSeconds，但是maxSeconds 好像官網寫棄用了
    logging.info("end prob.solve")


    # Extract results
    assignments = {(group, machine): pulp.value(x[group][machine]) for group in groups for machine in machines}
    fractions = {(group, machine): pulp.value(f[group][machine]) for group in groups for machine in machines}
    objective_value = pulp.value(prob.objective)

    return assignments, fractions, objective_value



def get_total_pipeline_latency(mywork, bottleneck_layer_idx):
    last_CONV_start_OU_cycle = 0 # 代表最後的 CONV layer 開始的時間，太小了直接設 0
    

    
    middle_OU_cycle_list = [ \
        (mywork.Tile_time[i]) * Config.NETWORK_DICT["OFM_row"][i] * Config.NETWORK_DICT["BLOCK_MIDDLE_TIME"][i] \
        for i in bottleneck_layer_idx ] # 因為 bottlneck layer 可能有多個，所以 bottleneck_layer_idx 是 list


    end_OU_cycle = 0 # 代表最後的 CONV layer 算完後的收尾，太小了直接設 0


    print(f" bottleneck_layer_idx ={bottleneck_layer_idx}")
    for i in bottleneck_layer_idx:
        OFM_row_string = "OFM_row"
        BLOCK_MIDDLE_TIME_string = "BLOCK_MIDDLE_TIME"
        print(f"mywork.Tile_time[i] = {mywork.Tile_time[i]}")
        print(f"Config.NETWORK_DICT[OFM_row][i] = {Config.NETWORK_DICT[OFM_row_string][i]}")
        print(f"Config.NETWORK_DICT[BLOCK_MIDDLE_TIME][i] = {Config.NETWORK_DICT[BLOCK_MIDDLE_TIME_string][i]}")

    # logger.info(f"bottleneck_layer_idx = {bottleneck_layer_idx}")
    # logger.info(f"middle_OU_cycle_list in total pipeline latency = {middle_OU_cycle_list}")
    middle_OU_cycle = max(middle_OU_cycle_list)


    total_pipeline_OU_cycle = last_CONV_start_OU_cycle + middle_OU_cycle + end_OU_cycle
    total_pipeline_latency = total_pipeline_OU_cycle *  mywork.OU
    return total_pipeline_latency



def find_bottleneck(mywork):


    # total area 13.95
    middle_OU_cycle_list = [ \
        (mywork.Tile_time[i]) * Config.NETWORK_DICT["OFM_row"][i] * Config.NETWORK_DICT["BLOCK_MIDDLE_TIME"][i] \
        for i in range(Config.NETWORK_DICT["total_layer_num"]) ] # 因為 bottlneck layer 可能有多個，所以 bottleneck_layer_idx 是 list
    # logger.info(f"middle_OU_cycle_list = {middle_OU_cycle_list}")
    bottleneck_OU_cycle = max(middle_OU_cycle_list)
    bottleneck_layer_idx = [i for i,ei in enumerate(middle_OU_cycle_list) if ei==bottleneck_OU_cycle]
    return bottleneck_layer_idx, bottleneck_OU_cycle
    


    # # total area 12.17
    # OU_cycle_of_each_layer = [ mywork.Tile_time[i] * Config.NETWORK_DICT["DELTA_X"][i]  for i in range(Config.NETWORK_DICT["total_layer_num"])]
    # bottleneck_OU_cycle = max(OU_cycle_of_each_layer)
    # bottleneck_layer_idx = [i for i,ei in enumerate(OU_cycle_of_each_layer) if ei==bottleneck_OU_cycle]
    # return bottleneck_layer_idx, bottleneck_OU_cycle



def add_Macro_and_rerun_parallelize_OU_to_Macro(mywork, bottleneck_layer_idx):
    for i in bottleneck_layer_idx:
        
        # 加 Macro 在有最大 PE_time 的那些 PE
        # 這樣子才能更新 Tile_time
        max_PE_time = max(mywork.PE_time[i])
        max_PE_time_PE_idx = [j for j,ej in enumerate(mywork.PE_time[i]) if ej == max_PE_time]
        for j in max_PE_time_PE_idx:
            # 加 Macro
            mywork.PE_num_Macro[i][j] = mywork.PE_num_Macro[i][j] + 1


            # 重跑 parallelize_OU_to_Macro
            if( mywork.PE_num_Macro[i][j] < 2 ): # 代表 mywork.num_PE_OU_shape[layer_idx][PE_idx] < mywork.num_OU_per_Macro, 也就是 OU shape 在 1個 Macro 內裝不下
                mywork.PE_time[i][j] = mywork.sum_of_PE_num_input_output_for_each_OU_shape[i][j] * Config.NETWORK_DICT["BIT_IFM"] / mywork.PE_num_Macro[i][j]
            else:
                # 正確版
                # assignments, fractions, mywork.PE_time[layer_idx][PE_idx] = parallelize_method.parallelize_OU_to_Macro( \
                #                                                             cycle_for_each_OU=PE_OU_cycle_for_each_OU_shape[PE_idx]\
                #                                                             ,num_Macro = mywork.PE_num_Macro[layer_idx][PE_idx]\
                #                                                             ,num_OU_per_Macro = mywork.num_OU_per_Macro)


                # 估計版，因為 parallelize_OU_to_Macro 太花時間
                mywork.PE_time[i][j] = mywork.sum_of_PE_num_input_output_for_each_OU_shape[i][j] * Config.NETWORK_DICT["BIT_IFM"] / mywork.PE_num_Macro[i][j]

        # 更新 Tile_time
        mywork.Tile_time[i] = max(mywork.PE_time[i])   




        
def reach_LATENCY(mywork):
    
    # 初始的 total pipeline latency
    bottleneck_layer_idx, bottleneck_OU_cycle = find_bottleneck(mywork)
    total_pipeline_latency = get_total_pipeline_latency(mywork, bottleneck_layer_idx)
    total_macro_num = sum([sum(mywork.PE_num_Macro[i]) * Config.NETWORK_DICT["K"][i]  for i in range(Config.NETWORK_DICT["total_layer_num"])])
    logger.info(f"      total latency = {total_pipeline_latency * Config.CLK_PERIOD / 1e+06}ms, Macro num = {total_macro_num}" )
    

    # latency 跟 加 Macro
    while(  total_pipeline_latency > Config.LATENCY ):
        
        
        

        # 在那些 bottleneck_layer_idx 加 Macro
        add_Macro_and_rerun_parallelize_OU_to_Macro(mywork, bottleneck_layer_idx)        


        # 找 bottleneck 在哪幾層 ( 可能有多個，所以是 list )
        bottleneck_layer_idx, bottleneck_OU_cycle = find_bottleneck(mywork)


        # total_pipeline_latency 更新
        total_pipeline_latency = get_total_pipeline_latency(mywork, bottleneck_layer_idx)
        total_macro_num = sum([sum(mywork.PE_num_Macro[i]) * Config.NETWORK_DICT["K"][i]  for i in range(Config.NETWORK_DICT["total_layer_num"])])
        logger.info(f"      total latency = {total_pipeline_latency * Config.CLK_PERIOD / 1e+06}ms, Macro num = {total_macro_num}")
        # logger.info(f"      PE_num_macro = {mywork.PE_num_Macro}") 
        # logger.info(f"      mywork.PE_time = {mywork.PE_time}")

       

        #
        bottleneck_latency = bottleneck_OU_cycle * mywork.OU


    return bottleneck_layer_idx, bottleneck_latency, total_pipeline_latency