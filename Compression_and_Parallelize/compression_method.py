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

def CoarseGrain():
    return

def FineGrain():
    return

def non_zero_counts(input):
    output = np.count_nonzero(input, axis=1)
    return output

def calc_Dist(input, OUY):
    # print("intput shape is",input.shape)
    # print("input type",input.dtype)
    c = np.zeros(input.shape[1],dtype=int) # centroid
    # print('centoird type',c.dtype)
    for i,ei in enumerate(input):
        c = np.bitwise_or(c, ei)
    count_ones = np.count_nonzero(c)
    OU_num = math.ceil(count_ones/OUY)
    return c, count_ones, OU_num

def calc_Dist_for_fine_grain(v, c):

    # below is tww slow
    '''
    summation = 0
    for i in range(c.shape[0]):
        summation = summation + ( v[i]**2 + c[i]**2 - v[i]*c[i] )
    '''

    # below is the same as calc_Dist
    '''
    c = np.bitwise_or(v,c)
    count_ones = np.count_nonzero(c)
    print("summation==count_ones",(summation==count_ones))
    '''


    result = v*v + c*c - v*c
    return(result.sum())

    
    '''
    c = [] # centroid
    for i in range(OUY):
        summation = sum(haha[i] for haha in input)
        dist = sum()
        
        value = summation / (OUX*2)
        c.append(value)
    return c,
    ''' 

def update_centroid_for_fine_grain(input, OUX, OUY):
    #below is too slow
    '''
    c = np.zeros(input.shape[1]) # centroid
    for i in range(input.shape[1]):
        summation = sum(haha[i] for haha in input)
        value = summation / (OUX*2)
        c[i] = value
        print("c[i] is",c[i])
    print()
    '''


    c = input.sum(axis=0)
    c = c / (OUX*2)

    c_for_count_ones = np.zeros(input.shape[1],dtype=int) # centroid
    for i in input:
        c_for_count_ones = np.bitwise_or(c_for_count_ones, i)
    count_ones = np.count_nonzero(c_for_count_ones)
    OU_num = math.ceil(count_ones/OUY)

    return c, c_for_count_ones, count_ones, OU_num




def Coarse_and_Fine(output_reshape, OUX, OUY):
    print(f"output_reshape for compression_algorithm shape is",output_reshape.shape)

    
    # 要 transpose => padding => multibit_per_cell => sort(會得出 output_packet)
    
    
    
    # transpose
    output_reshape = np.transpose(output_reshape, (1,0)) # 這邊有 transpose，後面才接 padding，ex. 把 CONV1 的 PE0 的 512x9 變 9x512

    
    

    # padding
    output_reshape = data_preprocess.padding_due_to_OUX(output_reshape=output_reshape, OUX=OUX) # 這邊是已經切割成 FullReuse 且經過 eliminate column 的 transpose，且 shape[0]=9，ex. CONV1 PE0 9x512, ex. OUX=16, 9x512 -> 16x512，


    # 做 bit_per_cell 合併，得到接下來的 output_reshape
    # ex. bit_per_cell=2
    # 10_10_01_00 => 1_1_1_0
    # 10_11_00_01 => 1_1_0_1
    # ex. CONV1 PE0 9x512 => 9x256
    output_reshape_original = output_reshape.copy() # group_after_CoarseFine 會需要使用，先存著
    output_reshape = data_preprocess.multibit_per_cell(output_reshape, Config.BIT_PER_CELL)
    

    # 檢查 shape
    # if count_CONV==0 :
    #     if not( output_reshape.shape[1]== 256 ): # 假設 weight 8-bit, bit_per_cell = 2
    #         print(f" something wrong with output_reshape shape ")
    #         print(f"output_reshape shape is {output_reshape.shape}")
    #         os.exit(0)
    


    # padding or not
    print("padding output_reshape data type", output_reshape.dtype)
    

    



    # count NZ for every filter
    output_NZ_count = np.count_nonzero(output_reshape, axis=1)
    # for j,ej in enumerate(output_NZ_count):
    #     print(f"output_NZ_count[{j}] : {ej}")
    # for j in output_reshape[0]:
    #     print(j,end='')


    # Count the occurrences of each value in output_NZ_count
    value_counts = np.bincount(output_NZ_count)
    show_value_counts = value_counts[np.min(output_NZ_count):] # ignore the #NZ that have no vectors
    sorted_indices = np.argsort(output_NZ_count) # sort from small to large number of nonzeroes, which means number of zeros from large to small
    # Sort output_NZ_count and output_reshape based on the sorted indices
    # sorted_output_NZ_count = output_NZ_count[sorted_indices]
    # sorted_output_reshape = output_reshape[sorted_indices]
    # print(f"sorted_indices : {sorted_indices}")
    # for i in sorted_indices:
    #     print(i)
    # print("sorted output NZ count")
    # for i in sorted_output_NZ_count:
    #     print(i)
    # print("sorted output reshape")
    # for i in sorted_output_reshape:
    #     print(i)




    # random sample for output_packet, or maybe can use stratified sampling
    # or in full case, output_packet = output_reshape
    # indices, mean, std, output_packet = random_sample(output_reshape, output_NZ_count, value_counts, sorted_indices, package_size)
    output_packet = output_reshape[sorted_indices]
    # print("NZ count is", np.count_nonzero(output_packet, axis=1))



    # can reverse output_packet back to output_reshape, later used in cluster_contain_what_indices_output_reshape
    original_indices = np.argsort(sorted_indices)
    # print(f"original_indices = {original_indices}")
    output_reshape_reversed = output_packet[original_indices]
    print(" output_reshape_reversed == output_reshape ",np.array_equal(output_reshape, output_reshape_reversed)) # check equal
    



    # some parameter
    arrayX = output_packet.shape[0]
    arrayY = output_packet.shape[1]
    package_size = arrayX
    # print("nz count", np.count_nonzero(output_packet, axis=1))


    # indices = [329, 134, 490, 271, 495, 194, 127, 473, 23, 257]
    # output_packet = output_reshape[indices]


    # check cluster_contain_what_indices_output_reshape
    # file_name = "/home/mark/k-means/cluster_contain_what_indices_output_reshape/" + "cluster_contain_what_indices_output_reshape_CONV" + str(count_CONV+1) + '_OU=' + str(OUX) + '.pickle'
    # print(f"load cluster_contain_what_indices_output_reshape from {file_name}")
    # with open(file_name, 'rb') as file:
    #     cluster_contain_what_indices_output_reshape = pickle.load(file)
    # for j, ej in enumerate(cluster_contain_what_indices_output_reshape):
    #     print(f"cluster_contain_what_indices_output_reshape[{j}]")
    #     print(f"    {ej}")
    # continue



    # check dictionary list 
    # file_name = "/home/mark/k-means/dictionary_list/" + "dictionary_list_CONV" + str(count_CONV+1) + '_OU=' + str(OUX) + '.pickle'
    # print(f"load dictionary_list from {file_name}")
    # with open(file_name, 'rb') as file:
    #     dictionary_list = pickle.load(file)
    # for j, ej in enumerate(dictionary_list):
    #     print(f"dictionary_list[{j}]")
    #     for key, value in dictionary_list[j].items():
    #         print(f"  key:{key}, how many:{len(value)} val:{value} ")
    # continue




    # # check OU_table
    # file_name = "/home/mark/k-means/OU_table/" + "OU_table_CONV" + str(count_CONV+1) + '_OU=' + str(OUX) + '.pickle'
    # print(f"load OU_table from {file_name}")
    # with open(file_name, 'rb') as file:
    #     OU_table = pickle.load(file)
    # print(f"len(OU_table) : {len(OU_table)}")
    # print(f"OU_table_CONV{str(count_CONV+1)}_OU={str(OUX)} : #Macro = { len(OU_table) / ((128/OUX)**2) }")
    # for key, value in OU_table.items():
    #     print(f"  key:{key}, how many:{len(value)} val:{value} ")
    # continue



    #################### Coarse Grain precentralize ####################
    # initial centroid
    c = np.zeros(( int(arrayX/2), arrayY ), dtype=int)
    c_group_num = np.full( ( int(arrayX/2) ), 2 )
    c2v = [[] for _ in range(int(arrayX/2))]  # Initialize c2v as a list
    for j in range(c.shape[0]):
        comb_pack = np.array([output_packet[j*2], output_packet[j*2+1]])
        c[j], _, _ = calc_Dist(comb_pack, OUY)
        c2v[j].extend([j*2, j*2+1]) 
    # c2v = np.array(c2v)
    # print("c2v is ",c2v)
    # print("c count NZ ", np.count_nonzero(c, axis=1)) 



    # below part is original coarse grain precentralize, 但為了跑得快先拿掉了
    '''
    # clustering
    iterations = 0
    max_iterations = 1
    group_list = [ ]
    # while (len(output_packet) !=0) and:
    # print("output packet is ")
    # print(output_packet)
    while( iterations < max_iterations):
        logging.info("iterations : %d",iterations)
        for j, ej in enumerate(output_packet): # itervate vector
            # print("i : ",i)
            # print("c2v is",c2v)
            # print("c_group_num is",c_group_num)
            # print("c nonzero count is",np.count_nonzero(c, axis=1))

            # tables of centroid with their closest ones 
            c_table = np.full((c.shape[0]), arrayY) # initialize
            c_table_count_ones = np.full((c.shape[0]), arrayY) # initialize
            for l, el in enumerate(c):
                if( c_group_num[l] != 0 and c_group_num[l] != OUX): # empty or already answer
                    for m, em in enumerate(c):
                        if( c_group_num[m] != 0 and l != m):
                            pack = np.array([el, em])
                            temp_c, temp_count, _ = calc_Dist( pack, OUY )
                            if(temp_count < c_table[l]):
                                c_table[l] = m
                                c_table_count_ones[l] = temp_count
                        else:
                            continue
                else:
                    continue
            # print("c_table, c_table_count_ones")
            # for j, (jc, jcco) in enumerate(zip(c_table, c_table_count_ones)): # check both same time
            #     print("j: ",j,"jc ",jc, "jcco ", jcco)

            # vector find nearest centroid
            closest_c = np.zeros(arrayY,dtype=int) # centroid
            closest_c_idx = 0
            min_count_ones = arrayY
            for l, el in enumerate(c):
                if(c_group_num[l]>0 and c_group_num[l]!=OUX): # not empty or not already answer
                    
                    
                    if(int(j/2)!=l): # not in my centroid, 這邊可以在自己的組內嗎？？？
                    # if(1==1):
                    
                    
                        temp_c, temp_count, OU_num = calc_Dist( np.array([ej, el]), OUY ) 
                        if(temp_count < min_count_ones): # vector finds closest centroid
                            closest_c = temp_c
                            closest_c_idx = l
                            min_count_ones = temp_count
                            # print("closest_c_idx, min_count_ones ", closest_c_idx, min_count_ones)
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            # UPDATE OR NOT !!!
            if(min_count_ones < c_table_count_ones[closest_c_idx]): # if cost is less, we group and update
                # print("closest_c, min_count_ones",closest_c, " ", min_count_ones)  
                
                # add vector
                c[closest_c_idx] = closest_c
                c2v[closest_c_idx].extend([j])
                c_group_num[closest_c_idx] += 1
                # print("add vector",np.count_nonzero(c[closest_c_idx]))
                
                # remove vector
                c2v[int(j/2)].remove(j) 
                c_group_num[int(j/2)] -= 1
                if(c_group_num[int(j/2)]==0): # centroid empty
                    c[int(j/2)] = np.full( (arrayY), 1 )
                elif(c_group_num[int(j/2)]==1): # one left
                    # print("left count ",c2v[int(i/2)][0])
                    c[int(j/2)] = output_packet[c2v[int(j/2)][0]]
                    # print("left count",np.count_nonzero(c[int(i/2)], axis=0)) 
                
                if(c_group_num[closest_c_idx] == OUX): # reach OUX size, can get one answer
                    # remember to remove the centroid answer
                    group_list.append((c2v[closest_c_idx], OU_num))
                    # print("got one answer ", c2v[j], " ", OU_num)
                    logging.info("got one answer %s,  %d", c2v[l], OU_num)
        iterations += 1


    # print("group_list is")
    # for i in group_list:
    #     print(i)
    # logging.info("group_list is %s", group_list)
    '''
    ############################################################



    ######################  fine grain fixed size ###############
    # logging.info("start fine grain fixed size")
    print("start fine grain fixed size")
    # choose centroids for fine grain
    k = math.ceil(arrayX / OUX)
    coarse_centroids = np.zeros((k, arrayY))
    coarse_centroids_indices = np.argsort((np.count_nonzero(c, axis=1))) # pick the most number of zeros
    # print("c is",c.shape)
    # print("coarse", coarse_centroids_indices)
    


    # pick coarse centroids
    # for i in range( k-len(group_list) ): # exclude the answer from coarse grain
    # print("k is",k)
    for j in range( k ): # don't exclude the answer from coarse grain yet
        # print("idx ",coarse_centroids_indices[j])
        coarse_centroids[j] = c[coarse_centroids_indices[j]]
        # print("coarse",j,"/",k,coarse_centroids[j])



    # from numpy.random import default_rng
    # rng = default_rng()
    # haha = rng.choice(arrayX, size=k, replace=False)
    # # haha = np.random.randint(arrayX, size=k)
    # print("haha",haha)
    # coarse_centroids = output_packet[haha]



    # start fine grain
    converged = False
    iterations = 0
    max_iterations = 10
    tol=1e-4
    min_OU_num = math.ceil(arrayY/OUY) * math.ceil(arrayX/OUX)
    Distance_table = np.zeros(k)
    OU_num_table = np.zeros(k)
    cluster_contain_what_ones = np.zeros((k, arrayY))
    cluster_contain_what_indices = np.zeros((k, OUX), dtype=int)
    
    
    while converged==False and iterations < max_iterations:
        logging.info("start fine grain iterations/max_iterations = %d/%d ",iterations,max_iterations)
        # print("start fine grain iterations/max_iterations = %d/%d ",iterations,max_iterations)
        
        
        print("output packet size",output_packet.shape)
        print("coarse centroid size",coarse_centroids.shape)
        # create cost matrix for HUNGARIAN ALGORITHM
        slots = OUX
        cost_matrix = np.zeros((arrayX, arrayX))
        for j,ej in enumerate(output_packet):
            for l,el in enumerate(coarse_centroids):
                temp_distance = calc_Dist_for_fine_grain(ej, el)
                # print("ej",ej,"el",el,"temp_distance",temp_distance)
                for m in range(slots): # same distance in the same cluster
                    cost_matrix[j][l*OUX+m] = temp_distance
        # print("cost_matrix", cost_matrix)


        # HUNGARIAN ALGORITHM
        logging.info("start HUNGARIAN")
        # print("start HUNGARIAN")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # print(f"row: {row_ind} col : {col_ind}")


        # update centroid
        logging.info("start update centroid")
        fine_centroids = np.zeros((k, arrayY))
        for j, ej in enumerate(coarse_centroids): # iterate every cluster
            # print("update cluster i ",i)

            # find which vector belong to centoird_i, beware if eliminate rows then would be which vector
            # if eliminate columns then would be which row
            new_fine_centroids_indices = np.where( col_ind//OUX == j )[0] 
            cluster_contain_what_indices[j] = new_fine_centroids_indices # record indices later for Digital Part Compression
            # print(f"new fine ind {new_fine_centroids_indices}")
            # print(f"cluster_contain_what_indices[{j}] : {cluster_contain_what_indices[j]}")
            pack = output_packet[new_fine_centroids_indices] # beware here is output_packet
            
            # hoho  = output_packet[np.ix_(new_fine_centroids_indices)]
            # for l, el in enumerate(hoho):
            #     print(f"hoho[{l}] : {el}")
            
            # for l,el in enumerate(pack):
            #     print(f"pack[{l}] : {el}")
            fine_centroids[j], cluster_contain_what_ones[j],Distance_table[j], OU_num_table[j] = update_centroid_for_fine_grain(pack, OUX, OUY)
            # print(f"cluster_contain_what_ones[{j}] : {cluster_contain_what_ones[j]}")
        #     print("OU_num_table",j," is ", OU_num_table[j])
        # print("min_OU_num_CoarseFine = ", OU_num_table.sum())

        
        # calculate the change in centroids and check for convergence
        # 現在有個問題是 update_centroid_for_fine_grain 那邊的 c[i] 是正常的
        # 但是 fine_centroids 都是 0，應該是因為 dtype=int 的關係
        # logging.info("start check converged")
        centroid_change = fine_centroids - coarse_centroids
        centroid_change_value = np.abs(centroid_change).max()
        # print("fine coarse change")
        # for j in range(k):
        #     print(j,"/",k,fine_centroids[j], coarse_centroids[j],centroid_change[j])

        # print("change is",fine_centroids - coarse_centroids)
        # print("centroid change is", centroid_change)
        converged = (centroid_change_value < tol)
        # logging.info("converged : %s", converged)
        print("converged : %s", converged)

        coarse_centroids = fine_centroids
        iterations += 1
    # print("min_OU_num",min_OU_num.sum())


    # can check with exhaustion
    # file_path_group = 'group_12x2x2.npy'
    # group = np.load(file_path_group, allow_pickle=True)
    # values = [x[1] for x in group]
    # for i in group:
    #     print(i)
    # print("max - min",np.max(values) - np.min(values))
    # print("exhaustion min_OU_num",np.min(values))


    # sum up every cluster
    min_OU_num_CoarseFine = OU_num_table.sum()

    

    # save pretty table
    # tb.add_rows([[package_size, OUX, OUY, indices, mean, std, min_OU_num_CoarseFine]])
    # tb.add_rows([[package_size, OUX, OUY, "NAN", "NAN", "NAN", min_OU_num_CoarseFine, "NAN"]])
    ############################################################


    return cluster_contain_what_ones, cluster_contain_what_indices, original_indices, output_reshape_original


def DigitalPart(cluster_contain_what_ones, cluster_contain_what_indices, original_indices, output_reshape_original, OU):
    ######################  Digital Part ###############
    logging.info("Digital Part Compression")
    # print("Digital Part Compression")


    # check cluster_contain_what and cluster_contain_what_indices
    # print(f"cluster_contain_what_ones is {cluster_contain_what_ones}")
    # print(f"cluster_contain_what_indices is {cluster_contain_what_indices}")
    # 
    # for j,ej in enumerate(cluster_contain_what):
    #     hey_indices = np.where(cluster_contain_what_ones[j]==1 ) 
    #     print(f"{j} : {hey_indices}")
    # for j,ej in enumerate(cluster_contain_what_ones[0]):
    #     print(f"{j} : {ej}")
    # os.exit(0)



    # categorize shapes in each cluster
    bitline_shape_of_each_cluster = []
    shape_union = set()
    for j, ej in enumerate(cluster_contain_what_indices): 
        list_A = np.where(cluster_contain_what_ones[j]==1)[0].tolist() # 用個 list A 存 np.where(cluster_contain_what_ones[j]==1) 
        list_B = cluster_contain_what_indices[j].tolist() # 用個 list B 存 cluster_contain_what_indices[j] 
        list_C = [l*Config.BIT_PER_CELL + k for l in list_A for k in range(Config.BIT_PER_CELL)] # ex. list_A = [0, 2], bit_per_cell=2, list_C = [0,1,4,5]
        # print(f"cluster_contain_what_ones[{j}] {cluster_contain_what_ones[j]}") #對的                        
        # print(f"cluster_contain_what_ones[{j}]_list_A : {list_A}") #對的
        # print(f"cluster_contain_what_ones[{j}]_list_C : {list_C}") 
        # print(f"list_B : {list_B}") # 對的
        
        
        
        # 下面先將 list_B 轉換成原本的 index，因為 list_B 是對應 output_packet，要找說 output_packet 的 list_B 是對應 output_reshape_original 的哪些 row
        # 注意是 output_packet[original_indices[list_B].tolist()] == output_reshape_original[list_B]
        # np.where(original_indices == b) 回傳 array, 要用兩次 [0][0] 才找到在 output_reshape_original 的位置
        # 再用 np.ix_ 切割 output_reshape_original，然後 
        # group_after_Coarsefine 會transpose 成 shape = len(list_C) * OU, 又 len(list_C) = len(list_A) * bit_per_cell
        # 然後為了 看哪些 column 的 shape 一樣，所以
        # ex.
        # row2 = 10100000  # row2 在上，因為 output_packet 把 0 多的往上排
        # row1 = 10110000
        # bit_percell =2
        # 會變成 ([1010, 1011, 0000, 0000])
        # 所以 group_after_CoarseFine 最後的 shape 是 len(list_A), bit_per_cell*OU 吧？？？
        group_after_CoarseFine = np.transpose( output_reshape_original[np.ix_( [np.where(original_indices == b)[0][0] for b in list_B], list_C)] )  # 還是有錯
        group_after_CoarseFine = group_after_CoarseFine.reshape(-1, Config.BIT_PER_CELL * group_after_CoarseFine.shape[1])
        
        
        

        

        
        # for l, el in enumerate(haha):
        #     print(f"haha[{l}]")
        #     for k, ek in enumerate(el):
        #         print(f"{ek}", end='')
        #     print()

        # if count_CONV==0 and j==3 and OUX==8: # 對的
        #     print(f"now list_B is {list_B}")
        #     hoho  = output_packet[np.ix_(list_B, list_A)]
        #     for l, el in enumerate(hoho):
        #         print(f"hoho[{l}] : {el}")
        #     os.exit(0)

        


        bitline_shape = dict()
        # print(f"group_after_CoarseFine[{j}: {group_after_CoarseFine[j]}")
        for l in range(len(list_A)):
            # print(f"el : {el}")
            el_tuple = tuple(group_after_CoarseFine[l])
            # print(f"el_tuple : {el_tuple}")

            # convert to decimal number
            binary_str = ''.join(map(str, el_tuple))
            decimal_number = int(binary_str, 2)

            '''
            if el_tuple in d:
                # print(f"ei tuple is {el_tuple}")
                d[el_tuple].append(l)
                
            else:
                # print(f"else ei tuple is {el_tuple}")
                d[el_tuple] = [l]
                shape_union.add(el_tuple)
            '''


            # d 的 key 是 shape，value 是哪些 vector，
            # 而且只記錄 list_A 的
            # 所以假設 bit_per_cell = 3
            # d[23] = [45,50,98]
            # 那其實是對應 vector 45*3, 45*3+1, 45*3+2, 50*3, 50*3+1......
            if decimal_number in bitline_shape:
                # print(f"ei tuple is {el_tuple}")
                bitline_shape[decimal_number].append(list_A[l])
            else:
                # print(f"else ei tuple is {el_tuple}")
                bitline_shape[decimal_number] = [list_A[l]]
                shape_union.add(decimal_number)


        # C 跑 column_compress
        bitline_shape_of_each_cluster.append(bitline_shape)
    

    # find the corresponding row in output_reshape from
    # using cluster_contain_what_indices_output_reshape
    # 因為現在的 cluster_contain_what_indices 是紀錄給 output_packet
    # OU_table 需要用到
    cluster_contain_what_indices_output_reshape = []
    for j,ej in enumerate(cluster_contain_what_indices):
        cluster_contain_what_indices_output_reshape.append( original_indices[cluster_contain_what_indices[j].tolist()] )
    # print("check cluster_contain_what_indices_output_reshape")
    # for j,ej in enumerate(cluster_contain_what_indices_output_reshape):
    #     print(f"    [{j}] : {ej}")
    # continue


    # 用 bitline_shape_of_each_cluster 做 OU_table, 去找歸類同樣的 shapes 組成可以組成同一個 OU
    OU_table = OU_Table.create_OU_table(cluster_contain_what_indices, bitline_shape_of_each_cluster, OU)

    



    # save pretty table
    # tb.add_rows([[package_size, OUX, OUY, which_PE, "NAN", "NAN", "NAN", min_OU_num_CoarseFine, len(OU_table), "NAN"]])
    

    
    # save dictionary_list to .pickle
    # file_name = "/home/mark/k-means/dictionary_list/" + "dictionary_list_CONV" + str(count_CONV+1) + '_OU=' + str(OUX) + '.pickle'
    # logging.info("dictionary_list to %s", file_name)
    # with open(file_name, 'wb') as file:
    #     pickle.dump(dictionary_list, file)




    # 紀錄 dictionary_list[j] 對應的哪些 row
    # 但為了 save cluster_contain_what_indices, but need to transfer from output_packet back to output_reshape
    # file_name = "/home/mark/k-means/cluster_contain_what_indices_output_reshape/" + "cluster_contain_what_indices_output_reshape_CONV" + str(count_CONV+1) + '_OU=' + str(OUX) + '.pickle'
    # logging.info("cluster_contain_what_indices_output_reshape to %s", file_name)
    # with open(file_name, 'wb') as file:
    #     pickle.dump(cluster_contain_what_indices_output_reshape, file)
    


    # save OU_table to .pickle
    # file_name = "/home/mark/k-meansB/OU_table_FullReuse/" + "OU_table_CONV" + str(count_CONV+1) + '_OU=' + str(OUX) + '_PE' + str(which_PE) + '.pickle'
    # logging.info("save OU_table to %s", file_name)
    # with open(file_name, 'wb') as file:
    #     pickle.dump(OU_table, file)
    
    # return OU_table, tb
    return OU_table
