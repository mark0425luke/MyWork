import sys
sys.path.append('..')

import Config
import math
import logging
import pickle
import MyWork
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
    num_cluster_types_for_each_OU_shape = []
    sum_of_num_cluster_types_for_each_OU_shape = 0
    num_input_output_for_each_OU_shape = []
    sum_of_num_input_output_for_each_OU_shape = 0
    num_OU_shape = len(OU_table)
    for key, value in OU_table.items(): # key 是 OU 的 shape, value 是 (input cluster->output vector)
        

        # logger.info(f"key = {key}")
        # logger.info(f"value = {value}")
        # ex.
        # shape 1 : (cluster1 -> ....), (cluster2 -> ....), (cluster4 -> ....)
        # shape 2 : (cluster1 -> ....), (cluster3 -> ....), (cluster4 -> ....), (cluster5 -> ....)
        # 則 num_input_for_each_OU = [3,4]
        # 而沒有 bitline 壓縮的話，可能就會變
        # ex. OU=2
        # shape 1 : (cluster1 -> v4, v5), (cluster1 -> v8, v50), (cluster2 -> v3, v6), (cluster4 -> v1, v7), (cluster4 -> ....), (cluster4 -> ....)
        # shape 2 : (cluster1 -> ....), (cluster1 -> ....), (cluster3 -> ....), (cluster3 -> ....), (cluster3 -> ....), (cluster4 -> ....), (cluster5 -> ....)
        # 則 num_types_cluster_for_each_OU_shape  = [3, 4]
        # 則 num_input_output_for_each_OU_shape = [6, 7]
        num_input_output_for_each_OU_shape.append(len(value))
        what_kind_of_cluster = set()
        for item in value:
            what_kind_of_cluster.add(tuple(item[0])) # item[0] 是 cluster, item[1] 是 output vectors
        num_cluster_types_for_each_OU_shape.append(len(what_kind_of_cluster))
        

        
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
    sum_of_num_cluster_types_for_each_OU_shape = sum(num_cluster_types_for_each_OU_shape)
    return MAX_NUM_FILTER, at_least_num_Macro, num_OU_shape,\
        num_cluster_types_for_each_OU_shape, sum_of_num_cluster_types_for_each_OU_shape,\
        num_input_output_for_each_OU_shape, sum_of_num_input_output_for_each_OU_shape



# ===============================================================
# 純粹為了不要引用 main 才在這邊建的 class
# =============================================================== 
class MyWork():
    def __init__(self, OU):
        self.OU = OU
        self.num_OU_per_Macro       = int( math.ceil( (Config.XB_paramX/self.OU)**2 ) )
 
        self.ADC_PRECISION          = (Config.BIT_PER_CELL + Config.BIT_DAC + math.log(self.OU, 2)) if (Config.BIT_PER_CELL!=1 and Config.BIT_DAC!=1) \
                                        else (Config.BIT_PER_CELL + Config.BIT_DAC + math.log(self.OU, 2)-1)


        self.PE_MAX_NUM_FILTER      = [ [0 for j in range(Config.NETWORK_DICT["K"][i])] for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Tile_MAX_NUM_FILTER    = [ 0 for j in range(Config.NETWORK_DICT["total_layer_num"])]


        self.PE_num_Macro   = [ [0 for j in range(Config.NETWORK_DICT["K"][i])] for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.num_Tile               = [ 0 for i in range(Config.NETWORK_DICT["total_layer_num"])] # 這個之後 Activation, Pooling, Rowbuffers 需要

        self.num_PE_OU_shape        = [  [0 for j in range(Config.NETWORK_DICT["K"][i])] for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.sum_of_PE_num_input_output_for_each_OU_shape = [  [0 for j in range(Config.NETWORK_DICT["K"][i])] for i in range(Config.NETWORK_DICT["total_layer_num"])]

        self.PE_time                = [[0 for i in range(Config.NETWORK_DICT["K"][i])] for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Tile_time              = [ 0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        

        

        # self.DUPLICATE              = [ 1 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.bottleneck_layer_idx = None
        self.BOTTLENECK_LATENCY     = 0 # 單位 : cycle
        self.total_pipeline_latency = None



        self.conv_area_model        = None
        self.conv_energy_model      = None

        self.pooling_area_Model     = None
        self.pooling_energy_Model   = None


        self.router_area_model      = None
        self.router_energy_model    = None

    def get_parameters(self):
        # print(f"mywork parameters : ")
        # print(f"        OU = {self.OU}")
        # print(f"        num_OU_per_Macro = {self.num_OU_per_Macro}")
        # print(f"        ADC_PRECISION = {self.ADC_PRECISION}")
        # print(f"        PE_MAX_NUM_FILTER = {self.PE_MAX_NUM_FILTER}")
        # print(f"        Tile_MAX_NUM_FILTER = {self.Tile_MAX_NUM_FILTER}")
        # print(f"        PE_at_least_num_Tile = {self.PE_at_least_num_Tile}")
        # print(f"        num_Tile = {self.num_Tile}")
        # print(f"        PE_time = {self.PE_time}")
        # print(f"        Tile_time = {self.Tile_time}")
        # print(f"        DUPLICATE = {self.DUPLICATE}")
        # print(f"        BOTTLENECK_LATENCY = {self.BOTTLENECK_LATENCY}")

        logging.info(f"mywork parameters : ")
        logging.info(f"        OU = {self.OU}")
        logging.info(f"        num_OU_per_Macro = {self.num_OU_per_Macro}")
        logging.info(f"        ADC_PRECISION = {self.ADC_PRECISION}")
        logging.info(f"        PE_MAX_NUM_FILTER = {self.PE_MAX_NUM_FILTER}")
        logging.info(f"        Tile_MAX_NUM_FILTER = {self.Tile_MAX_NUM_FILTER}")
        logging.info(f"        PE_num_Macro = {self.PE_num_Macro}")
        logging.info(f"        num_Tile = {self.num_Tile}")
        logging.info(f"        num_PE_OU_shape = {self.num_PE_OU_shape}")
        logging.info(f"        sum_of_PE_num_input_output_for_each_OU_shape = {self.sum_of_PE_num_input_output_for_each_OU_shape}")
        logging.info(f"        PE_time = {self.PE_time}")
        logging.info(f"        Tile_time = {self.Tile_time}")
        # logging.info(f"        DUPLICATE = {self.DUPLICATE}")
        logging.info(f"        BOTTLENECK_LATENCY = {self.BOTTLENECK_LATENCY}")
        logging.info(f"        total_pipeline_latency = {self.total_pipeline_latency}")
############################################




if __name__=='__main__':
    
    # ===============================================================
    # 設定
    # =============================================================== 
    layer_idx = 10
    PE_idx = 0
    ############################################



    # ===============================================================
    # 讀取 mywork
    # =============================================================== 
    # 讀 pickle
    file_name   =  Config.home_dir + "mywork_after_parallelize_OU_to_Macro/" \
                + 'OU=' + str(Config.OU) + '_BIT_PER_CELL=' + str(Config.BIT_PER_CELL) \
                + '_required_LATENCY=' + str(Config.LATENCY*Config.CLK_PERIOD/1e+06) + 'ms'\
                + '.pickle'
    print(f"load : {file_name}")
    with open(file_name, 'rb') as file:
        mywork = pickle.load(file)  
    ############################################
    

    
    
    # ===============================================================
    # 讀取 OU table
    # =============================================================== 
    
    file_name   = Config.home_dir + "OU_table_FullReuse_without_bitline_compression/" \
                + "OU_table_CONV" + str(layer_idx+1) \
                + '_OU=' + str(Config.OU) + '_BIT_PER_CELL=' + str(Config.BIT_PER_CELL) \
                + '_PE' + str(PE_idx) \
                + '.pickle'
    print(f"load : {file_name}")
    try:
        # Open the pickle file in binary read mode
        with open(file_name, 'rb') as file:
            # Load the data from the pickle file
            PE_OU_table = pickle.load(file)
    # Now, `loaded_OU_table` contains the Python object stored in the pickle file
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except EOFError:
        print("Error: Ran out of input while unpickling (possibly empty or corrupted pickle file).")
    except Exception as e:
        print(f"An error occurred: {e}")
    ############################################


    print(f"PE_OU_table key is= {PE_OU_table.keys()}")
    
    # ===============================================================
    # 印出參數
    # =============================================================== 
    PE_MAX_NUM_FILTER, PE_at_least_num_Macro, PE_num_OU_shape,\
    num_cluster_types_for_each_OU_shape, sum_of_num_cluster_types_for_each_OU_shape,\
    num_input_output_for_each_OU_shape, sum_of_num_input_output_for_each_OU_shape = get_OU_table_parameter(PE_OU_table, mywork)
    
    print(f"PE_MAX_NUM_FILTER   =   {PE_MAX_NUM_FILTER}")
    print(f"PE_at_least_num_Macro   =   {PE_at_least_num_Macro}")
    print(f"PE_num_OU_shape =   {PE_num_OU_shape}")
    print(f"num_cluster_types_for_each_OU_shape =   {num_cluster_types_for_each_OU_shape}")
    print(f"sum_of_num_cluster_types_for_each_OU_shape  =   {sum_of_num_cluster_types_for_each_OU_shape}")
    print(f"num_input_output_for_each_OU_shape  =   {num_input_output_for_each_OU_shape}")
    print(f"sum_of_num_input_output_for_each_OU_shape   =   {sum_of_num_input_output_for_each_OU_shape}")
    ############################################