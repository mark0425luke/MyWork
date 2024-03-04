import math
import Config
import math
import logging
import pickle

logger = logging.getLogger()

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
        self.sum_of_PE_num_cluster_types_for_each_OU_shape = [  [0 for j in range(Config.NETWORK_DICT["K"][i])] for i in range(Config.NETWORK_DICT["total_layer_num"])]
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
        logging.info(f"        sum_of_PE_num_cluster_types_for_each_OU_shape = {self.sum_of_PE_num_cluster_types_for_each_OU_shape}")
        logging.info(f"        sum_of_PE_num_input_output_for_each_OU_shape = {self.sum_of_PE_num_input_output_for_each_OU_shape}")
        logging.info(f"        PE_time = {self.PE_time}")
        logging.info(f"        Tile_time = {self.Tile_time}")
        # logging.info(f"        DUPLICATE = {self.DUPLICATE}")
        logging.info(f"        BOTTLENECK_LATENCY = {self.BOTTLENECK_LATENCY}")
        logging.info(f"        total_pipeline_latency = {self.total_pipeline_latency}")
        


