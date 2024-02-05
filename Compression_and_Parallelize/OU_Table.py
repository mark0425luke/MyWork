import sys
sys.path.append('..')

import Config
import math
import logging
import pickle
logger = logging.getLogger()

def sort_the_key_of_each_cluster(dictionary_list):
    for i, ei in enumerate(dictionary_list):
        dictionary_list[i] = dict(sorted(dictionary_list[i].items())) 

def create_OU_table(cluster_contain_what_indices, bitline_shape_of_each_cluster, OU):

    # 這邊要做的就是把 bitline_shape_of_each_cluster( 裡面裝 cluster 個 dictionary ) 給做成 OU table
    # 而且用的是沒有 bitline 壓縮版本
    # ex. OU=2, bitline shape 有 a,b,c,d ....
    # cluster 1 的 dictionary 是
    #       (key, value) =
    #            (c, [v7,v8,v9,v18])
    #            (a, [v1, v5, v6, v10, v13])
    #            (b, [v2,v3,v4])
    # cluster 2 的 dictionary 是
    #       (key, value) =
    #            (c, [v13, v28, v39, v66])
    #            (a, [v4, v5, v6, v7, v8])
    #            (b, [v0, v1, v3, v9, v10])
    # 想要做成
    # OU_table 是
    #     (key : value) = 
    #         (aa, :  (cluster1, [[v1],[v5]]), 
    #                 (cluster1, [[v6],[v10]]), 
    #                 (cluster2, [[v4],[v5]]),
    #                 (cluster2, [[v6],[v7]]))
    #         (ab  :  (cluster1, [[v13],[v2]])
    #                 (cluster2, [[v8],[v0]])),
    #         (bb  :  (cluster1, [[v2],[v3]]),
    #                 (cluster2, [[v0],[v1]]),
    #                 (cluster2, [[v3],[v9]]),),
    #         ......
    # 之所以一個 vector 就要是自己一個 list 是因為之後要檢查 MAX_NUM_FILTER 是不是等於 1
    #
    # 1. 先 sort 每個 cluster 的 key 之後容易找到一樣的 OU shape
    # cluster 1 的 dictionary 變成
    #       (key, value) =
    #            (a, [v1, v5, v6, v10, v13])
    #            (b, [v2,v3,v4])
    #            (c, [v7,v8,v9,v18])
    # cluster 2 的 dictionary 是
    #       (key, value) =
    #            (a, [v4, v5, v6, v7, v8])
    #            (b, [v0, v1, v3, v9, v10])
    #            (c, [v13, v28, v39, v66])
    # 
    # 2. iterate 各個 cluster 的 dictionary，把他們攤平成
    # a  a  a  a    a  b  b  b .....
    # v1 v5 v6 v10 v13 v2 v3 v4....
    #
    # 3. 每次取 OU 個當作 shape 並得到 output vector
    # 然後添加到 OU_table
    # OU_table[aa].append( (cluster1, [[v1],[v5]]) )
    # OU_table[aa].append( (cluster1, [[v6],[v10]]) )
    # ......

    

    # 1. sort 每個 cluster 的 dictionary 的 key
    sort_the_key_of_each_cluster(bitline_shape_of_each_cluster)


    #
    OU_table = dict()
    for i, ei in enumerate(bitline_shape_of_each_cluster):
        # 2. 攤平
        flattened_dic = flatten_dictionary(bitline_shape_of_each_cluster[i])
        
        # 3. 添加到 OU_table
        append_to_OU_table(cluster_contain_what_indices, i, flattened_dic, OU, OU_table)

    return OU_table


    
def flatten_dictionary(dic):
    """ Flatten the dictionary into a list of [key, value] pairs. """
    return [[key, value] for key, values in dic.items() for value in values]

def append_to_OU_table(cluster_contain_what_indices, cluster_idx, flattened, OU, OU_table):
    for i in range(0, len(flattened), OU):
        # Extract the current chunk
        chunk = flattened[i : i+OU]

        # Create the key for OU_table (e.g., 'aa', 'ab', etc.)
        ou_key = tuple(item[0] for item in chunk)
        
        # Prepare the values to append, 也就是 output vectors
        # ex. OU=2 會得出 [[v4], [v5]]
        values_to_append = [[item[1]] for item in chunk]


        # Append to OU_table
        if ou_key in OU_table:
            OU_table[ou_key].append((cluster_contain_what_indices[cluster_idx], values_to_append))
        else:
            OU_table[ou_key] = [(cluster_contain_what_indices[cluster_idx], values_to_append)]


def get_OU_table_parameter(OU_table, mywork):
    
    MAX_NUM_FILTER = 0
    at_least_num_Macro = int( math.ceil( len(OU_table) / (mywork.num_OU_per_Macro)) ) # 也就是 (幾個 shape) / (一個Macro多少Ou * 一個 Tile 多少 Macro)
    num_input_output_for_each_OU_shape = []
    sum_of_num_input_output_for_each_OU_shape = 0
    num_OU_shape = len(OU_table)
    for key, value in OU_table.items(): # key 是 OU 的 shape, value 是 (input cluster->output vector)
        
        # logger.info(f"key = {key}")
        # logger.info(f"value = {value}")
        # ex.
        # shape 1 : (cluster1 -> ....), (cluster2 -> ....), (cluster4 -> ....)
        # shape 1 : (cluster1 -> ....), (cluster3 -> ....), (cluster4 -> ....), (cluster5 -> ....)
        # 則 num_input_for_each_OU = [3,4]
        # 而沒有 bitline 壓縮的話，可能就會變
        # ex. OU=2
        # shape 1 : (cluster1 -> v4, v5), (cluster1 -> v8, v50), (cluster2 -> v3, v6), (cluster4 -> v1, v7), (cluster4 -> ....), (cluster4 -> ....)....
        # shape 1 : (cluster1 -> ....), (cluster1 -> ....), (cluster3 -> ....), (cluster3 -> ....), (cluster3 -> ....), (cluster4 -> ....), (cluster5 -> ....)
        # 則 num_input_for_each_OU = [6, 7]
        num_input_output_for_each_OU_shape.append(len(value))

        
        # ex.BIT_W=8, BIT_PER_CELL = 1
        # value[0] = (cluster1, [[v1, v2, v5], [v3, v18], [.....], [.....]]) # ex. OU=4, 4 bitline
        # value[0][1] = [[v1, v2, v5], [v3, v18], [.....], [.....]]
        # so there are 4 bitline, each bitline represent vectors that are compressed to 1 bitline
        # MAX_NUM_FILTER is calculated according to each bitline and to find which bitline contains the most filter
        # 當 BIT_W=8, BIT_PER_CELL = 3 時， 
        # v0 表 F[8:6] ( 其中 F[8] 是補位 ), 所以 v0, v1, v2 是 F0
        # v3, v4, v5 是 F1
        for i in range(len(value)): # iterlate every (input cluster->output vector)
            for bitline in value[i][1]: # 取每個 bitline 有的 vector
                
                
                # ex. BIT_W=8, BIT_PER_CELL = 3,
                # bitline = [v0, v1, v4, v5, F6]
                # bitline_has_what_filter = [F0, F0, F1, F1, F2]
                # tmp_NUM_FILTER = 3
                bitline_has_what_filters = [x//(math.ceil(Config.NETWORK_DICT["BIT_W"] / Config.BIT_PER_CELL)) for x in bitline]
                tmp_NUM_FILTER = len(set(bitline_has_what_filters)) 
                
                
                MAX_NUM_FILTER = tmp_NUM_FILTER if(tmp_NUM_FILTER > MAX_NUM_FILTER) else MAX_NUM_FILTER

    sum_of_num_input_output_for_each_OU_shape = sum(num_input_output_for_each_OU_shape)
    return MAX_NUM_FILTER, at_least_num_Macro, num_input_output_for_each_OU_shape, sum_of_num_input_output_for_each_OU_shape, num_OU_shape


def get_PE_OU_cycle_for_each_OU_shape(layer_idx, OU, PE_idx):

    # 存 PE_cycle
    file_name   = "/home/mark/MyWork_OU2/PE_OU_cycle_for_each_OU_shape/" \
                    + "CONV" + str(layer_idx+1) \
                    + '/OU=' + str(mywork.OU) \
                    + '_num_Macro_per_Tile=' + str(Config.num_Macro_per_Tile)\
                    + '_PE' + str(PE_idx) \
                    + '.pickle'
    with open(file_name, 'rb') as file:
        PE_OU_cycle_for_each_OU_shape = pickle.load(file) 


        

    print(f"CONV{layer_idx+1}, OU={OU}, PE_idx={PE_idx}, PE_OU_cycle_for_each_OU_shape = {PE_OU_cycle_for_each_OU_shape}")

if __name__=='__main__':
    print("haha")
    get_PE_OU_cycle_for_each_OU_shape(layer_idx=12, OU=4, PE_idx=0)