from Hardware_Analysis import Hardware_Analysis
from Compression_and_Parallelize import compression_method
from Compression_and_Parallelize import OU_Table
from Compression_and_Parallelize import parallelize_method
from Compression_and_Parallelize import data_preprocess
import Config
import math
import pickle
import logging

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
        




def main():
    # logfile 設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Prints log messages to the terminal
            logging.FileHandler('./Report/logfile.txt', mode='w')  # Saves log messages to a file
        ]
    )

    # home directory
    home_dir = "/home/mark/MyWork_OU2/"




    #
    logging.info(f"Config OU = {Config.OU}")
    mywork = MyWork(OU=Config.OU)

    '''
    # ===============================================================
    # Coarse, Fine, DigitalPart 跟 OU_table
    # ===============================================================
    for layer_idx in range(Config.NETWORK_DICT["total_layer_num"]):
        logging.info(f"     running CONV{layer_idx+1}")


        
        # 讀 binary weight 跟 切割給每個 PE
        weight_for_PE = data_preprocess.data_preprocess(layer_idx)
        
        # 收集每個  PE 的 OU_table
        # Tile_OU_table = [None for i in range(Config.NETWORK_DICT["K"][layer_idx])]

        # CoarseFine 跟 Digital Part 得到 OU_table
        for K_idx in range(Config.NETWORK_DICT["K"][layer_idx]):
            
            
            # CoarseFine
            cluster_contain_what_ones, cluster_contain_what_indices, original_indices, output_reshape_original \
                = compression_method.Coarse_and_Fine(output_reshape=weight_for_PE[K_idx], OUX=mywork.OU, OUY=mywork.OU)
            
            
            
            # Digital Part, 得到 OU_table
            tmp_OU_table = compression_method.DigitalPart(cluster_contain_what_ones, cluster_contain_what_indices, original_indices, output_reshape_original, OU=mywork.OU)
            
            
            
            # 存 OU_table 
            file_name   = home_dir + "OU_table_FullReuse_without_bitline_compression/" \
                        + "OU_table_CONV" + str(layer_idx+1) \
                        + '_OU=' + str(mywork.OU) + '_BIT_PER_CELL=' + str(Config.BIT_PER_CELL) \
                        + '_PE' + str(K_idx) \
                        + '.pickle'
            with open(file_name, 'wb') as file:
                pickle.dump(tmp_OU_table, file)  
            logging.info(f"dump {file_name}")


    # os.exit(0)
    ############################################
    '''
    
    
    
    '''
    # ===============================================================
    # 讀取 OU table, 得到初始的 Tile_time, num_Macro
    # ===============================================================
    for layer_idx in range(Config.NETWORK_DICT["total_layer_num"]):
        PE_num_input_output_for_each_OU_shape = [ [] for j in range(Config.NETWORK_DICT["K"][layer_idx]) ]
        PE_OU_cycle_for_each_OU_shape = [ [] for j in range(Config.NETWORK_DICT["K"][layer_idx]) ]
        
        for PE_idx in range(0, Config.NETWORK_DICT["K"][layer_idx]):


            # 讀取 OU_table 
            file_name   = home_dir + "OU_table_FullReuse_without_bitline_compression/" \
                        + "OU_table_CONV" + str(layer_idx+1) \
                        + '_OU=' + str(mywork.OU) + '_BIT_PER_CELL=' + str(Config.BIT_PER_CELL) \
                        + '_PE' + str(PE_idx) \
                        + '.pickle'
            logging.info(f"load : {file_name}")
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


            # 提取 OU table 參數
            mywork.PE_MAX_NUM_FILTER[layer_idx][PE_idx],  \
            mywork.PE_num_Macro[layer_idx][PE_idx],  \
            PE_num_input_output_for_each_OU_shape[PE_idx], \
            mywork.sum_of_PE_num_input_output_for_each_OU_shape[layer_idx][PE_idx], \
            mywork.num_PE_OU_shape[layer_idx][PE_idx] = OU_Table.get_OU_table_parameter(PE_OU_table, mywork)

            
            # 得到初始的 PE_time
            logging.info(f"CONV{layer_idx+1} PE{PE_idx} ready to run parallelize_OU_to_Macro")
            PE_OU_cycle_for_each_OU_shape[PE_idx] = [ Config.NETWORK_DICT["BIT_IFM"] * num_input for num_input in PE_num_input_output_for_each_OU_shape[PE_idx]]
            if( mywork.PE_num_Macro[layer_idx][PE_idx] < 2 ): # 代表 mywork.num_PE_OU_shape[layer_idx][PE_idx] < mywork.num_OU_per_Macro, 也就是 OU shape 在 1個 Macro 內裝不下
                logging.info(f"     mywork.num_PE_OU_shape[layer_idx][PE_idx] = {mywork.num_PE_OU_shape[layer_idx][PE_idx]}")
                logging.info(f"     mywork.num_OU_per_Macro = {mywork.num_OU_per_Macro}")
                logging.info(f"     since mywork.num_PE_OU_shape[layer_idx][PE_idx] < mywork.num_OU_per_Macro")
                logging.info(f"     no need to run parallelize_OU_to_Macro")
                mywork.PE_time[layer_idx][PE_idx] = mywork.sum_of_PE_num_input_output_for_each_OU_shape[layer_idx][PE_idx] * Config.NETWORK_DICT["BIT_IFM"] / mywork.PE_num_Macro[layer_idx][PE_idx]
            else:
                # 正確版
                # assignments, fractions, mywork.PE_time[layer_idx][PE_idx] = parallelize_method.parallelize_OU_to_Macro( \
                #                                                             cycle_for_each_OU=PE_OU_cycle_for_each_OU_shape[PE_idx]\
                #                                                             ,num_Macro = mywork.PE_num_Macro[layer_idx][PE_idx]\
                #                                                             ,num_OU_per_Macro = mywork.num_OU_per_Macro)

                # 估計版，因為 parallelize_OU_to_Macro 太花時間
                mywork.PE_time[layer_idx][PE_idx] = mywork.sum_of_PE_num_input_output_for_each_OU_shape[layer_idx][PE_idx] * Config.NETWORK_DICT["BIT_IFM"] / mywork.PE_num_Macro[layer_idx][PE_idx]
        
        # 更新 Tile_time
        mywork.Tile_time[layer_idx] = max(mywork.PE_time[layer_idx])    
        
        
        
        # 這邊可以檢查 Bitline 壓縮取消後是不是都是 1
        mywork.Tile_MAX_NUM_FILTER[layer_idx]   = max(mywork.PE_MAX_NUM_FILTER[layer_idx])
        if mywork.Tile_MAX_NUM_FILTER[layer_idx] !=1:
            logging.info(f" Digital Part is wrong")
            os.exit(0)

    
    ############################################
    
    
    

    
    # ===============================================================
    # total_pipeline_latency 達標 跟 加 Macro
    # ===============================================================


    # total_pipeline_latency 達標
    logging.info(f"start reach_LATENCY")
    mywork.bottleneck_layer_idx, mywork.BOTTLENECK_LATENCY, mywork.total_pipeline_latency = parallelize_method.reach_LATENCY(mywork)
    
    # total pipeline latency 達標後, 才切 Tile
    for layer_idx in range(Config.NETWORK_DICT["total_layer_num"]):
        # 原版，每 16 個 Macro 切一個 Tile
        # mywork.num_Tile[layer_idx]              = max( [ math.ceil(tmp_num_Macro/Config.MAX_num_Macro_per_Tile) for tmp_num_Macro in mywork.PE_num_Macro[layer_idx]] )

        # 除以 BIT_IFM 版
        mywork.num_Tile[layer_idx]              = max( [ math.ceil(math.ceil(tmp_num_Macro/Config.NETWORK_DICT["BIT_IFM"]) / Config.MAX_num_Macro_per_Tile) for tmp_num_Macro in mywork.PE_num_Macro[layer_idx]] )


    # 存一下 total_pipeline_latency 達標後的 mywork, 參數有 OU, BIT_PER_CELL, total_pipeline_latency 
    file_name   =  home_dir + "mywork_after_parallelize_OU_to_Macro/" \
                + 'OU=' + str(mywork.OU) + '_BIT_PER_CELL=' + str(Config.BIT_PER_CELL) \
                + '_required_LATENCY=' + str(Config.LATENCY*Config.CLK_PERIOD/1e+06) + 'ms'\
                + '.pickle'
    with open(file_name, 'wb') as file:
        pickle.dump(mywork, file)  
    logging.info(f"dump {file_name}")
    
    ############################################
    '''
   


    
                
        
   

    # ===============================================================
    # Hardware Analysis
    # ===============================================================

    # 讀 pickle
    file_name   =  home_dir + "mywork_after_parallelize_OU_to_Macro/" \
                + 'OU=' + str(mywork.OU) + '_BIT_PER_CELL=' + str(Config.BIT_PER_CELL) \
                + '_required_LATENCY=' + str(Config.LATENCY*Config.CLK_PERIOD/1e+06) + 'ms'\
                + '.pickle'
    with open(file_name, 'rb') as file:
        mywork = pickle.load(file)  
    


    # cheat_Tile_time = [ \
    #     (Config.NETWORK_DICT["K"][i] * Config.NETWORK_DICT["BIT_W"] * Config.NETWORK_DICT["OUT_CH"][i] * Config.NETWORK_DICT["IN_CH"][i]) \
    #     * 0.1 / (Config.num_Macro_per_Tile * mywork.OU**2) for i in range(Config.NETWORK_DICT["total_layer_num"])]
    # mywork.Tile_time = cheat_Tile_time
    # mywork.Tile_time = [3,	62.5,	72,	147.5,	157,	310,	317,	330.5,	677,	674.5,	674,	665,	656]



    #
    Hardware_Analysis.Hardware_Analysis(mywork)

    #
    mywork.get_parameters()
    
    


    #
    # logging.info(f"num_Macro_per_Tile = {Config.num_Macro_per_Tile}")
    logging.info(f"MAX_num_Macro_per_Tile = {Config.MAX_num_Macro_per_Tile}")
    logging.info(f" only CONV energy   = {sum(mywork.conv_energy_model.total_energy_CONV)* (1e-09) * (1e+03)}mJ")
    logging.info(f" only Pooling energy   = {sum(mywork.pooling_energy_Model.Pooling_energy )* (1e-09) * (1e+03)}mJ")
    logging.info(f" only Pooling energy of each layer : {mywork.pooling_energy_Model.Pooling_energy}")
    logging.info(f" only Router energy   = {mywork.router_energy_model.total_energy_Router * (1e-09) * (1e+03)}mJ")
    logging.info(f" only CONV energy of each layer")
    for i in range(Config.NETWORK_DICT["total_layer_num"]):
        logging.info(f"CONV{i+1} :  ratio = {mywork.conv_energy_model.total_energy_CONV[i] / sum(mywork.conv_energy_model.total_energy_CONV)}, energy = {mywork.conv_energy_model.total_energy_CONV[i]* (1e-09) * (1e+03)}mJ")

    logging.info(f" only CONV area     = {sum(mywork.conv_area_model.total_area_CONV) * (1e-06)}mm^2")
    logging.info(f" only Pooling area     = {sum(mywork.pooling_area_Model.Pooling_area) * (1e-06)}mm^2")
    logging.info(f" only Router area     = {mywork.router_area_model.total_area_Router * (1e-06)}mm^2")

    logging.info(f" only Macro num              = {sum(mywork.conv_area_model.Macro_num)}")
    logging.info(f" only Macro area     = {sum(mywork.conv_area_model.Macro_area) * (1e-06)}mm^2")
    
    logging.info(f" only OU_table which OU              Macro num     = {sum(mywork.conv_area_model.which_OU_num) }")
    logging.info(f" only OU_table cluster input         Macro num     = {sum(mywork.conv_area_model.cluster_input_num)}")
    logging.info(f" only OU_table weight_bit_position   Macro num     = {sum(mywork.conv_area_model.weight_bit_position_num) }")
    logging.info(f" only OU_table which_filter          Macro num     = {sum(mywork.conv_area_model.which_filter_num)}")
    logging.info(f" only OU_table total                 Macro num     = {sum(mywork.conv_area_model.which_OU_num) + sum(mywork.conv_area_model.cluster_input_num) + sum(mywork.conv_area_model.weight_bit_position_num) + sum(mywork.conv_area_model.which_filter_num)}")

    logging.info(f" only ADC num = {sum(mywork.conv_area_model.ADC_num)}")


#     下面都要補上 除以 BIT_IFM 後的版本
#     sum_of_HW_power = 0
#     for i in range(Config.NETWORK_DICT["total_layer_num"]):
#         logging.info(f"CONV{i+1}")
#         sum_of_HW_power +=\
#         (mywork.conv_area_model.Activation_num[i] * Config.Activation[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], 29)]["power"] + \
#         mywork.conv_area_model.Row_Buffers_num[i] * Config.CLK_PERIOD * Config.Row_Buffers[(Config.CLK_PERIOD, Config.NETWORK_DICT["SRAM_NUM_WORD"][i], Config.NETWORK_DICT["SRAM_WIDTH"][i])]["power"] + \
#         mywork.conv_area_model.which_OU_num[i]              * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"] +\
#         mywork.conv_area_model.cluster_input_num[i]         * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]  +\
#         mywork.conv_area_model.weight_bit_position_num[i]   * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]     +\
#         mywork.conv_area_model.which_filter_num[i]          * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]      +\
#         mywork.conv_area_model.Macro_num[i] * Config.Macro[(0)]["device_power"] * (mywork.OU)**2 + \
#         mywork.conv_area_model.SandH_num[i] * Config.SandH[(0)]["power"] + \
#         mywork.conv_area_model.ADC_num[i]* Config.ADC[(mywork.ADC_PRECISION, 1.28)]["power"]           + \
#         mywork.conv_area_model.Shift_and_Add_num[i]* Config.Shift_and_Add[(Config.CLK_PERIOD, Config.BIT_PER_CELL, Config.BIT_DAC, mywork.OU, Config.NETWORK_DICT["BIT_W"],  Config.NETWORK_DICT["BIT_IFM"])]["power"] + \
#         mywork.conv_area_model.Decoder_num[i] * Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["power"] + \
#         mywork.conv_area_model.MUX_num[i] * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"] + \
#         # mywork.conv_area_model.Add_num[i]* Config.Add_base[(Config.CLK_PERIOD, Config.Add_and_Distributor_Small_input, 21, 29)]["power"])
#         mywork.conv_area_model.Add_num[i]* Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["power"])
# 
#         Activation_HW_power = mywork.conv_area_model.Activation_num[i] * Config.Activation[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], 29)]["power"]
#         Row_Buffers_HW_power = mywork.conv_area_model.Row_Buffers_num[i] * Config.CLK_PERIOD * Config.Row_Buffers[(Config.CLK_PERIOD, Config.NETWORK_DICT["SRAM_NUM_WORD"][i], Config.NETWORK_DICT["SRAM_WIDTH"][i])]["power"] 
#         which_OU_HW_power = mywork.conv_area_model.which_OU_num[i]              * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"] 
#         cluster_input_HW_power = mywork.conv_area_model.cluster_input_num[i]         * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"] 
#         weight_bit_position_HW_power = mywork.conv_area_model.weight_bit_position_num[i]   * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"] 
#         which_filter_HW_power = mywork.conv_area_model.which_filter_num[i]          * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]      
#         Macro_HW_power = mywork.conv_area_model.Macro_num[i] * Config.Macro[(0)]["device_power"] * (mywork.OU)**2 
#         SandH_HW_power = mywork.conv_area_model.SandH_num[i] * Config.SandH[(0)]["power"] 
#         ADC_HW_power = mywork.conv_area_model.ADC_num[i]* Config.ADC[(mywork.ADC_PRECISION, 1.28)]["power"]        
#         Shift_and_Add_HW_power = mywork.conv_area_model.Shift_and_Add_num[i]* Config.Shift_and_Add[(Config.CLK_PERIOD, Config.BIT_PER_CELL, Config.BIT_DAC, mywork.OU, Config.NETWORK_DICT["BIT_W"],  Config.NETWORK_DICT["BIT_IFM"])]["power"]
#         Decoder_HW_power = mywork.conv_area_model.Decoder_num[i] * Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["power"] 
#         MUX_HW_power = mywork.conv_area_model.MUX_num[i] * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"] 
#         # Add_HW_power = mywork.conv_area_model.Add_num[i]* Config.Add_base[(Config.CLK_PERIOD, Config.Add_and_Distributor_Small_input, 21, 29)]["power"]
#         Add_HW_power = mywork.conv_area_model.Add_num[i]* Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["power"]
# 
# 
#         print(f"                                # * power                       vs       energy / Latency   ")
#         print(f" Activation              = {Activation_HW_power:.5f}            vs  {mywork.conv_energy_model.Activation_energy[i]          / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" Row_Buffers             = {Row_Buffers_HW_power:.5f}           vs  {mywork.conv_energy_model.Row_Buffers_energy[i]         / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" which_OU                = {which_OU_HW_power:.5f}              vs  {mywork.conv_energy_model.which_OU_energy[i]            / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" cluster_input           = {cluster_input_HW_power:.5f}         vs  {mywork.conv_energy_model.cluster_input_energy[i]       / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" weight_bit_position     = {weight_bit_position_HW_power:.5f}   vs  {mywork.conv_energy_model.weight_bit_position_energy[i] / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" which_filter            = {which_filter_HW_power:.5f}          vs  {mywork.conv_energy_model.which_filter_energy[i]        / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" Macro                   = {Macro_HW_power:.5f}                 vs  {mywork.conv_energy_model.OU_energy[i]                  / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" SandH                   = {SandH_HW_power:.5f}                 vs  {mywork.conv_energy_model.SandH_energy[i]               / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" ADC                     = {ADC_HW_power:.5f}                   vs  {mywork.conv_energy_model.ADC_energy[i]                 / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" Shift_and_Add           = {Shift_and_Add_HW_power:.5f}         vs  {mywork.conv_energy_model.Shift_and_Add_energy[i]       / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" Decoder                 = {Decoder_HW_power:.5f}               vs  {mywork.conv_energy_model.Decoder_energy[i]             / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" MUX                     = {MUX_HW_power:.5f}                   vs  {mywork.conv_energy_model.MUX_energy[i]                 / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#         print(f" Add                     = {Add_HW_power:.5f}                   vs  {mywork.conv_energy_model.Add_energy[i]                 / (mywork.total_pipeline_latency * Config.CLK_PERIOD)}")
#     print(f" sum of every hardware power  without router and poolinng = {sum_of_HW_power} W")
#     print(f" power calculated by total energy without router and pooling / (total_pipeline_latency * Config.CLK_PERIOD) = {(sum(mywork.conv_energy_model.total_energy_CONV))/ (mywork.total_pipeline_latency * Config.CLK_PERIOD) } W")
# 


    #
    logging.info(f"Number ======")
    logging.info(f"        Activation :             = {sum(mywork.conv_area_model.Activation_num)}")
    logging.info(f"        Pooling :                = {sum(mywork.pooling_area_Model.Pooling_num)}")
    logging.info(f"        Row_Buffers              = {sum(mywork.conv_area_model.Row_Buffers_num)         }")
    logging.info(f"        which_OU                 = {sum(mywork.conv_area_model.which_OU_num)            }")
    logging.info(f"        cluster_input            = {sum(mywork.conv_area_model.cluster_input_num)       }")
    logging.info(f"        weight_bit_position      = {sum(mywork.conv_area_model.weight_bit_position_num) }")
    logging.info(f"        which_filter             = {sum(mywork.conv_area_model.which_filter_num)        }")
    logging.info(f"        Macro                    = {sum(mywork.conv_area_model.Macro_num)               }")
    logging.info(f"        SandH                    = {sum(mywork.conv_area_model.SandH_num)               }")
    logging.info(f"        ADC                      = {sum(mywork.conv_area_model.ADC_num)                 }")
    logging.info(f"        Shift_and_Add            = {sum(mywork.conv_area_model.Shift_and_Add_num)       }")
    logging.info(f"        Decoder                  = {sum(mywork.conv_area_model.Decoder_num)             }")
    logging.info(f"        MUX_of_Shift_and_Add     = {sum(mywork.conv_area_model.MUX_of_Shift_and_Add_num)}")
    logging.info(f"        Adder_mask               = {sum(mywork.conv_area_model.Adder_mask_num)          }")
    logging.info(f"        Add                      = {sum(mywork.conv_area_model.Add_num)                 }")
    logging.info(f"        MUX_of_filter            = {sum(mywork.conv_area_model.MUX_of_filter_num)       }")
    logging.info(f"        Accumulator              = {sum(mywork.conv_area_model.Accumulator_num)         }")
    logging.info(f"        Router                   ={mywork.router_area_model.inter_CONV_Router_num + sum(mywork.router_area_model.intra_CONV_Router_num)}")
    

    #
    logging.info(f"AREA ======")
    for i in range(Config.NETWORK_DICT["total_layer_num"]):
        logging.info(f" CONV{i+1}: ")
        logging.info(f"        Activation_ratio            = {mywork.conv_area_model.Activation_area[i]             / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        Row_Buffers_ratio           = {mywork.conv_area_model.Row_Buffers_area[i]            / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        which_OU_ratio              = {mywork.conv_area_model.which_OU_area[i]               / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        cluster_input_ratio         = {mywork.conv_area_model.cluster_input_area[i]          / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        weight_bit_position_ratio   = {mywork.conv_area_model.weight_bit_position_area[i]    / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        which_filter_ratio          = {mywork.conv_area_model.which_filter_area[i]           / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        Macro_ratio                 = {mywork.conv_area_model.Macro_area[i]                  / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        SandH_ratio                 = {mywork.conv_area_model.SandH_area[i]                  / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        ADC_ratio                   = {mywork.conv_area_model.ADC_area[i]                    / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        Shift_and_Add_ratio         = {mywork.conv_area_model.Shift_and_Add_area[i]          / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        Decoder_ratio               = {mywork.conv_area_model.Decoder_area[i]                / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        MUX_of_Shift_and_Add_ratio  = {mywork.conv_area_model.MUX_of_Shift_and_Add_area[i]   / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        Adder_mask_ratio            = {mywork.conv_area_model.Adder_mask_area[i]             / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        Add_ratio                   = {mywork.conv_area_model.Add_area[i]                    / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        MUX_of_filter_ratio         = {mywork.conv_area_model.MUX_of_filter_area[i]          / mywork.conv_area_model.total_area_CONV[i]:.4f}")
        logging.info(f"        Accumulator_ratio           = {mywork.conv_area_model.Accumulator_area[i]            / mywork.conv_area_model.total_area_CONV[i]:.4f}")



    total_area = sum(mywork.conv_area_model.total_area_CONV) + sum(mywork.pooling_area_Model.Pooling_area) + mywork.router_area_model.total_area_Router
    sum_of_Activation_area              = sum( [mywork.conv_area_model.Activation_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_Row_Buffers_area             = sum( [mywork.conv_area_model.Row_Buffers_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_which_OU_area                = sum( [mywork.conv_area_model.which_OU_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_cluster_input_area           = sum( [mywork.conv_area_model.cluster_input_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_weight_bit_position_area     = sum( [mywork.conv_area_model.weight_bit_position_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_which_filter_area            = sum( [mywork.conv_area_model.which_filter_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_Macro_area                   = sum( [mywork.conv_area_model.Macro_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_SandH_area                   = sum( [mywork.conv_area_model.SandH_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_ADC_area                     = sum( [mywork.conv_area_model.ADC_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_Shift_and_Add_area           = sum( [mywork.conv_area_model.Shift_and_Add_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_Decoder_area                 = sum( [mywork.conv_area_model.Decoder_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_MUX_of_Shift_and_Add_area    = sum( [mywork.conv_area_model.MUX_of_Shift_and_Add_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_Adder_mask_area              = sum( [mywork.conv_area_model.Adder_mask_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_Add_area                     = sum( [mywork.conv_area_model.Add_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_MUX_of_filter_area           = sum( [mywork.conv_area_model.MUX_of_filter_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )
    sum_of_Accumulator_area             = sum( [mywork.conv_area_model.Accumulator_area[i] for i in range(Config.NETWORK_DICT["total_layer_num"])] )


    sum_of_Activation_area_ratio              = sum_of_Activation_area / total_area
    sum_of_Row_Buffers_area_ratio             = sum_of_Row_Buffers_area / total_area
    sum_of_which_OU_area_ratio                = sum_of_which_OU_area / total_area
    sum_of_cluster_input_area_ratio           = sum_of_cluster_input_area / total_area
    sum_of_weight_bit_position_area_ratio     = sum_of_weight_bit_position_area / total_area
    sum_of_which_filter_area_ratio            = sum_of_which_filter_area / total_area
    sum_of_Macro_area_ratio                   = sum_of_Macro_area / total_area
    sum_of_SandH_area_ratio                   = sum_of_SandH_area / total_area
    sum_of_ADC_area_ratio                     = sum_of_ADC_area / total_area
    sum_of_Shift_and_Add_area_ratio           = sum_of_Shift_and_Add_area / total_area
    sum_of_Decoder_area_ratio                 = sum_of_Decoder_area / total_area
    sum_of_MUX_of_Shift_and_Add_area_ratio    = sum_of_MUX_of_Shift_and_Add_area / total_area
    sum_of_Adder_mask_area_ratio              = sum_of_Adder_mask_area / total_area
    sum_of_Add_area_ratio                     = sum_of_Add_area / total_area
    sum_of_MUX_of_filter_area_ratio           = sum_of_MUX_of_filter_area / total_area
    sum_of_Accumulator_area_ratio             = sum_of_Accumulator_area / total_area
    sum_of_Pooling_area_ratio                 = sum(mywork.pooling_area_Model.Pooling_area)/ total_area
    sum_of_Router_area_ratio                  = mywork.router_area_model.total_area_Router / total_area

    sum_of_area_ratio = \
        + sum_of_Activation_area_ratio \
        + sum_of_Row_Buffers_area_ratio \
        + sum_of_which_OU_area_ratio \
        + sum_of_cluster_input_area_ratio \
        + sum_of_weight_bit_position_area_ratio \
        + sum_of_which_filter_area_ratio \
        + sum_of_Macro_area_ratio \
        + sum_of_SandH_area_ratio \
        + sum_of_ADC_area_ratio \
        + sum_of_Shift_and_Add_area_ratio \
        + sum_of_Decoder_area_ratio \
        + sum_of_MUX_of_Shift_and_Add_area_ratio \
        + sum_of_Adder_mask_area_ratio \
        + sum_of_Add_area_ratio \
        + sum_of_MUX_of_filter_area_ratio\
        + sum_of_Accumulator_area_ratio \
        + sum_of_Pooling_area_ratio \
        + sum_of_Router_area_ratio 
    logging.info(f"sum_of_area_ratio = {sum_of_area_ratio}")


    sum_of_area = \
        + sum_of_Activation_area \
        + sum_of_Row_Buffers_area \
        + sum_of_which_OU_area \
        + sum_of_cluster_input_area \
        + sum_of_weight_bit_position_area \
        + sum_of_which_filter_area \
        + sum_of_Macro_area \
        + sum_of_SandH_area \
        + sum_of_ADC_area \
        + sum_of_Shift_and_Add_area \
        + sum_of_Decoder_area \
        + sum_of_MUX_of_Shift_and_Add_area \
        + sum_of_Adder_mask_area \
        + sum_of_Add_area \
        + sum_of_MUX_of_filter_area \
        + sum_of_Accumulator_area \
        + sum(mywork.pooling_area_Model.Pooling_area) \
        + mywork.router_area_model.total_area_Router
    logging.info(f"sum_of_area = {sum_of_area}")
    logging.info(f"total area = {total_area/1e+06}mm^2")
    
    
    logging.info(f"total area ratio")
    logging.info(f"        Activation_ratio            = {sum_of_Activation_area_ratio:.3f}")
    logging.info(f"        Row_Buffers_ratio           = {sum_of_Row_Buffers_area / total_area:.3f}")
    logging.info(f"        OU_table_ratio              = {(sum_of_which_OU_area_ratio +sum_of_cluster_input_area_ratio +sum_of_weight_bit_position_area_ratio +sum_of_which_filter_area_ratio):.3f}")
    logging.info(f"        Macro_ratio                 = {sum_of_Macro_area_ratio:.3f}")
    logging.info(f"        SandH_ratio                 = {sum_of_SandH_area_ratio:.3f}")
    logging.info(f"        ADC_ratio                   = {sum_of_ADC_area_ratio:.3f}")
    logging.info(f"        Shift_and_Add_ratio         = {sum_of_Shift_and_Add_area_ratio:.3f}")
    logging.info(f"        Decoder_ratio               = {sum_of_Decoder_area_ratio:.3f}")
    logging.info(f"        MUX_of_Shift_and_Add_ratio  = {sum_of_MUX_of_Shift_and_Add_area_ratio:.3f}")
    logging.info(f"        Adder_mask_ratio            = {sum_of_Adder_mask_area_ratio:.3f}")
    logging.info(f"        Add_ratio                   = {sum_of_Add_area_ratio:.3f}")
    logging.info(f"        MUX_of_filter_ratio         = {sum_of_MUX_of_filter_area_ratio:.3f}")
    logging.info(f"        Accumulator_ratio           = {sum_of_Accumulator_area_ratio:.3f}")
    logging.info(f"        Pooling_ratio               = {sum_of_Pooling_area_ratio:.3f}")
    logging.info(f"        Router_ratio                = {sum_of_Router_area_ratio:.3f}")


 
    #
    logging.info(f"ENERGY ======")
    for i in range(Config.NETWORK_DICT["total_layer_num"]):
        logging.info(f" CONV{i+1}: ")
        logging.info(f"        Activation_ratio            = {mywork.conv_energy_model.Activation_energy[i]             / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        Row_Buffers_ratio           = {mywork.conv_energy_model.Row_Buffers_energy[i]            / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        which_OU_ratio              = {mywork.conv_energy_model.which_OU_energy[i]               / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        cluster_input_ratio         = {mywork.conv_energy_model.cluster_input_energy[i]          / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        weight_bit_position_ratio   = {mywork.conv_energy_model.weight_bit_position_energy[i]    / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        which_filter_ratio          = {mywork.conv_energy_model.which_filter_energy[i]           / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        OU_ratio                    = {mywork.conv_energy_model.OU_energy[i]                     / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        SandH_ratio                 = {mywork.conv_energy_model.SandH_energy[i]                  / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        ADC_ratio                   = {mywork.conv_energy_model.ADC_energy[i]                    / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        Shift_and_Add_ratio         = {mywork.conv_energy_model.Shift_and_Add_energy[i]          / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        Decoder_ratio               = {mywork.conv_energy_model.Decoder_energy[i]                / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        MUX_of_Shift_and_Add_ratio  = {mywork.conv_energy_model.MUX_of_Shift_and_Add_energy[i]   / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        Adder_mask_ratio            = {mywork.conv_energy_model.Adder_mask_energy[i]             / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        Add_ratio                   = {mywork.conv_energy_model.Add_energy[i]                    / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        MUX_of_filter_ratio         = {mywork.conv_energy_model.MUX_of_filter_energy[i]          / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")
        logging.info(f"        Accumulator_ratio           = {mywork.conv_energy_model.Accumulator_energy[i]            / mywork.conv_energy_model.total_energy_CONV[i]:.4f}")


    total_energy = sum(mywork.conv_energy_model.total_energy_CONV) + sum(mywork.pooling_energy_Model.Pooling_energy) + mywork.router_energy_model.total_energy_Router
    sum_of_Activation_energy              = sum(mywork.conv_energy_model.Activation_energy)
    sum_of_Row_Buffers_energy             = sum(mywork.conv_energy_model.Row_Buffers_energy)
    sum_of_which_OU_energy                = sum(mywork.conv_energy_model.which_OU_energy)
    sum_of_cluster_input_energy           = sum(mywork.conv_energy_model.cluster_input_energy)
    sum_of_weight_bit_position_energy     = sum(mywork.conv_energy_model.weight_bit_position_energy)
    sum_of_which_filter_energy            = sum(mywork.conv_energy_model.which_filter_energy)
    sum_of_Macro_energy                   = sum(mywork.conv_energy_model.OU_energy)
    sum_of_SandH_energy                   = sum(mywork.conv_energy_model.SandH_energy)
    sum_of_ADC_energy                     = sum(mywork.conv_energy_model.ADC_energy)
    sum_of_Shift_and_Add_energy           = sum(mywork.conv_energy_model.Shift_and_Add_energy)
    sum_of_Decoder_energy                 = sum(mywork.conv_energy_model.Decoder_energy)
    sum_of_MUX_of_Shift_and_Add_energy    = sum(mywork.conv_energy_model.MUX_of_Shift_and_Add_energy)
    sum_of_Adder_mask_energy              = sum(mywork.conv_energy_model.Adder_mask_energy)
    sum_of_MUX_of_filter_energy           = sum(mywork.conv_energy_model.MUX_of_filter_energy)
    sum_of_Add_energy                     = sum(mywork.conv_energy_model.Add_energy)
    sum_of_Accumulator_energy             = sum(mywork.conv_energy_model.Accumulator_energy)
    
    logging.info(f" sum of ADC switch cycle = {sum(mywork.conv_energy_model.ADC_switch_cycle)}")
    logging.info(f" each layer ADC switch cycle = {mywork.conv_energy_model.ADC_switch_cycle}")
    logging.info(f" sum of ADC energy = {sum_of_ADC_energy}")
    
    
    paper_ADC_switch_cycle = [ math.ceil(4*Config.NETWORK_DICT["OUT_CH"][i]) * math.ceil(Config.NETWORK_DICT["IN_CH"][i]/128) \
            * Config.NETWORK_DICT["K"][i]**2 * Config.NETWORK_DICT["OFM_row"][i]**2 * 8 for i in range(Config.NETWORK_DICT["total_layer_num"])]    
    logging.info(f" each layer paper_ADC_switch_cycle = {paper_ADC_switch_cycle}")
    logging.info(f"sum of ADC switch cycle of Optimizing Weight Mapping adc energy = { sum(paper_ADC_switch_cycle) } ")
    logging.info(f"Optimizing Weight Mapping adc energy = { sum(paper_ADC_switch_cycle) * 22.45 * 1e-09 } mJ")


    sum_of_Activation_energy_ratio              = sum_of_Activation_energy / total_energy
    sum_of_Row_Buffers_energy_ratio             = sum_of_Row_Buffers_energy / total_energy
    sum_of_which_OU_energy_ratio                = sum_of_which_OU_energy / total_energy
    sum_of_cluster_input_energy_ratio           = sum_of_cluster_input_energy / total_energy
    sum_of_weight_bit_position_energy_ratio     = sum_of_weight_bit_position_energy / total_energy
    sum_of_which_filter_energy_ratio            = sum_of_which_filter_energy / total_energy
    sum_of_Macro_energy_ratio                   = sum_of_Macro_energy / total_energy
    sum_of_SandH_energy_ratio                   = sum_of_SandH_energy / total_energy
    sum_of_ADC_energy_ratio                     = sum_of_ADC_energy / total_energy
    sum_of_Shift_and_Add_energy_ratio           = sum_of_Shift_and_Add_energy / total_energy
    sum_of_Decoder_energy_ratio                 = sum_of_Decoder_energy / total_energy
    sum_of_MUX_of_Shift_and_Add_energy_ratio    = sum_of_MUX_of_Shift_and_Add_energy / total_energy
    sum_of_Adder_mask_energy_ratio              = sum_of_Adder_mask_energy / total_energy
    sum_of_Add_energy_ratio                     = sum_of_Add_energy / total_energy
    sum_of_MUX_of_filter_energy_ratio           = sum_of_MUX_of_filter_energy / total_energy
    sum_of_Accumulator_energy_ratio             = sum_of_Accumulator_energy / total_energy
    sum_of_Pooling_energy_ratio                 = sum(mywork.pooling_energy_Model.Pooling_energy)/ total_energy
    sum_of_Router_energy_ratio                  = mywork.router_energy_model.total_energy_Router / total_energy

    sum_of_energy_ratio = \
        + sum_of_Activation_energy_ratio \
        + sum_of_Row_Buffers_energy_ratio \
        + sum_of_which_OU_energy_ratio \
        + sum_of_cluster_input_energy_ratio \
        + sum_of_weight_bit_position_energy_ratio \
        + sum_of_which_filter_energy_ratio \
        + sum_of_Macro_energy_ratio \
        + sum_of_SandH_energy_ratio \
        + sum_of_ADC_energy_ratio \
        + sum_of_Shift_and_Add_energy_ratio \
        + sum_of_Decoder_energy_ratio \
        + sum_of_MUX_of_Shift_and_Add_energy_ratio \
        + sum_of_Adder_mask_energy_ratio \
        + sum_of_Add_energy_ratio \
        + sum_of_MUX_of_filter_energy_ratio\
        + sum_of_Accumulator_energy_ratio \
        + sum_of_Pooling_energy_ratio \
        + sum_of_Router_energy_ratio 
    logging.info(f"sum_of_energy_ratio = {sum_of_energy_ratio}")


    sum_of_energy = \
        + sum_of_Activation_energy \
        + sum_of_Row_Buffers_energy \
        + sum_of_which_OU_energy \
        + sum_of_cluster_input_energy \
        + sum_of_weight_bit_position_energy \
        + sum_of_which_filter_energy \
        + sum_of_Macro_energy \
        + sum_of_SandH_energy \
        + sum_of_ADC_energy \
        + sum_of_Shift_and_Add_energy \
        + sum_of_Decoder_energy \
        + sum_of_MUX_of_Shift_and_Add_energy \
        + sum_of_Adder_mask_energy \
        + sum_of_Add_energy \
        + sum_of_MUX_of_filter_energy \
        + sum_of_Accumulator_energy \
        + sum(mywork.pooling_energy_Model.Pooling_energy) \
        + mywork.router_energy_model.total_energy_Router
    logging.info(f"sum_of_energy = {sum_of_energy}")
    logging.info(f"total energy = {total_energy}")

    
    
    logging.info(f"total energy ratio")
    logging.info(f"        Activation_ratio            = {sum_of_Activation_energy / total_energy:.3f}")
    logging.info(f"        Row_Buffers_ratio           = {sum_of_Row_Buffers_energy / total_energy:.3f}")
    logging.info(f"        OU_table_ratio              = {(sum_of_which_OU_energy + sum_of_cluster_input_energy + sum_of_weight_bit_position_energy + sum_of_which_filter_energy) / total_energy:.3f}")
    logging.info(f"        Macro_ratio                 = {sum_of_Macro_energy / total_energy:.3f}")
    logging.info(f"        SandH_ratio                 = {sum_of_SandH_energy / total_energy:.3f}")
    logging.info(f"        ADC_ratio                   = {sum_of_ADC_energy / total_energy:.3f}")
    logging.info(f"        Shift_and_Add_ratio         = {sum_of_Shift_and_Add_energy / total_energy:.3f}")
    logging.info(f"        Decoder_ratio               = {sum_of_Decoder_energy / total_energy:.3f}")
    logging.info(f"        MUX_of_Shift_and_Add_ratio  = {sum_of_MUX_of_Shift_and_Add_energy_ratio:.3f}")
    logging.info(f"        Adder_mask_ratio            = {sum_of_Adder_mask_energy / total_energy:.3f}")
    logging.info(f"        Add_ratio                   = {sum_of_Add_energy / total_energy:.3f}")
    logging.info(f"        MUX_of_filter_ratio         = {sum_of_MUX_of_filter_energy_ratio:.3f}")
    logging.info(f"        Accumulator_ratio           = {sum_of_Accumulator_energy / total_energy:.3f}")
    logging.info(f"        Pooling_ratio               = {sum(mywork.pooling_energy_Model.Pooling_energy)/ total_energy:.3f}")
    logging.info(f"        Router_ratio                = {mywork.router_energy_model.total_energy_Router / total_energy:.3f}")
    
    

    #
    logging.info(f"SWITCH CYCLE ======")
        
    
    sum_of_Activation_switch_cycle              = sum(mywork.conv_energy_model.Activation_switch_cycle)
    sum_of_Row_Buffers_switch_cycle             = sum(mywork.conv_energy_model.Row_Buffers_switch_cycle)
    sum_of_which_OU_switch_cycle                = sum(mywork.conv_energy_model.which_OU_switch_cycle)
    sum_of_cluster_input_switch_cycle           = sum(mywork.conv_energy_model.cluster_input_switch_cycle)
    sum_of_weight_bit_position_switch_cycle     = sum(mywork.conv_energy_model.weight_bit_position_switch_cycle)
    sum_of_which_filter_switch_cycle            = sum(mywork.conv_energy_model.which_filter_switch_cycle)
    sum_of_Macro_switch_cycle                   = sum(mywork.conv_energy_model.OU_switch_cycle)
    sum_of_SandH_switch_cycle                   = sum(mywork.conv_energy_model.SandH_switch_cycle)
    sum_of_ADC_switch_cycle                     = sum(mywork.conv_energy_model.ADC_switch_cycle)
    sum_of_Shift_and_Add_switch_cycle           = sum(mywork.conv_energy_model.Shift_and_Add_switch_cycle)
    sum_of_Add_and_Distributor_switch_cycle     = sum(mywork.conv_energy_model.Decoder_switch_cycle)
    sum_of_Adder_mask_switch_cycle              = sum(mywork.conv_energy_model.Adder_mask_switch_cycle)
    sum_of_Accumulator_switch_cycle             = sum(mywork.conv_energy_model.Accumulator_switch_cycle)
    
    conv_total_switch_cycle  \
    = sum_of_Activation_switch_cycle \
    + sum_of_Row_Buffers_switch_cycle \
    + sum_of_which_OU_switch_cycle \
    + sum_of_cluster_input_switch_cycle \
    + sum_of_weight_bit_position_switch_cycle \
    + sum_of_which_filter_switch_cycle \
    + sum_of_Macro_switch_cycle \
    + sum_of_SandH_switch_cycle \
    + sum_of_ADC_switch_cycle \
    + sum_of_Shift_and_Add_switch_cycle \
    + sum_of_Add_and_Distributor_switch_cycle
    + sum_of_Adder_mask_switch_cycle \
    + sum_of_Accumulator_switch_cycle

    total_switch_cycle = conv_total_switch_cycle + sum(mywork.pooling_energy_Model.Pooling_switch_cycle) + mywork.router_energy_model.Router_switch_cycle


    sum_of_Activation_switch_cycle_ratio              = sum_of_Activation_switch_cycle / total_switch_cycle
    sum_of_Row_Buffers_switch_cycle_ratio             = sum_of_Row_Buffers_switch_cycle / total_switch_cycle
    sum_of_which_OU_switch_cycle_ratio                = sum_of_which_OU_switch_cycle / total_switch_cycle
    sum_of_cluster_input_switch_cycle_ratio           = sum_of_cluster_input_switch_cycle / total_switch_cycle
    sum_of_weight_bit_position_switch_cycle_ratio     = sum_of_weight_bit_position_switch_cycle / total_switch_cycle
    sum_of_which_filter_switch_cycle_ratio            = sum_of_which_filter_switch_cycle / total_switch_cycle
    sum_of_Macro_switch_cycle_ratio                   = sum_of_Macro_switch_cycle / total_switch_cycle
    sum_of_SandH_switch_cycle_ratio                   = sum_of_SandH_switch_cycle / total_switch_cycle
    sum_of_ADC_switch_cycle_ratio                     = sum_of_ADC_switch_cycle / total_switch_cycle
    sum_of_Shift_and_Add_switch_cycle_ratio           = sum_of_Shift_and_Add_switch_cycle / total_switch_cycle
    sum_of_Add_and_Distributor_switch_cycle_ratio     = sum_of_Add_and_Distributor_switch_cycle / total_switch_cycle
    sum_of_Adder_mask_switch_cycle_ratio              = sum_of_Adder_mask_switch_cycle / total_switch_cycle
    sum_of_Accumulator_switch_cycle_ratio             = sum_of_Accumulator_switch_cycle / total_switch_cycle
    sum_of_Pooling_switch_cycle_ratio                 = sum(mywork.pooling_energy_Model.Pooling_switch_cycle)/ total_switch_cycle
    sum_of_Router_switch_cycle_ratio                  = mywork.router_energy_model.Router_switch_cycle / total_switch_cycle

    
    
    logging.info(f"total switch_cycle ratio")
    logging.info(f"        Activation_ratio            = {sum_of_Activation_switch_cycle_ratio:.6f}")
    logging.info(f"        Row_Buffers_ratio           = {sum_of_Row_Buffers_switch_cycle / total_switch_cycle:.6f}")
    logging.info(f"        OU_table_ratio              = {(sum_of_which_OU_switch_cycle_ratio +sum_of_cluster_input_switch_cycle_ratio +sum_of_weight_bit_position_switch_cycle_ratio +sum_of_which_filter_switch_cycle_ratio):.6f}")
    logging.info(f"        Macro_ratio                 = {sum_of_Macro_switch_cycle_ratio:.6f}")
    logging.info(f"        SandH_ratio                 = {sum_of_SandH_switch_cycle_ratio:.6f}")
    logging.info(f"        ADC_ratio                   = {sum_of_ADC_switch_cycle_ratio:.6f}")
    logging.info(f"        Shift_and_Add_ratio         = {sum_of_Shift_and_Add_switch_cycle_ratio:.6f}")
    logging.info(f"        Add_and_Distributor         = {sum_of_Add_and_Distributor_switch_cycle_ratio:.6f}")
    logging.info(f"        Adder_mask_ratio            = {sum_of_Adder_mask_switch_cycle_ratio:.6f}")
    logging.info(f"        Accumulator_ratio           = {sum_of_Accumulator_switch_cycle_ratio:.6f}")
    logging.info(f"        Pooling_ratio               = {sum_of_Pooling_switch_cycle_ratio:.6f}")
    logging.info(f"        Router_ratio                = {sum_of_Router_switch_cycle_ratio:.6f}")




    #
    # num_operations 公式計算
    # num_operations_of_each_layer = [Config.NETWORK_DICT["OUT_CH"][i] * Config.NETWORK_DICT["IN_CH"][i] * Config.NETWORK_DICT["K"][i]**2 \
    #     * Config.NETWORK_DICT["OFM_row"][i]**2 * 2 for i in range(Config.NETWORK_DICT["total_layer_num"])]
    # num_operations = sum( num_operations_of_each_layer )
    # logging.info(f"num_operations of each layer = {num_operations_of_each_layer}")
    
    
    # Optimizing Weight Mapping 回推
    # energy efficiency = throughput / power
    # 或是 energy efficiency = 多少OPs / energy
    # 4.767 (TOPs/s/W)  = 4767GOPs = 多少OPs / 0.00758J
    # 多少OPs = 36.13386 GOPs
    num_operations =  36.13386 * 1e+09 

    logging.info(f"energy efficiency = num_operations / total energy")
    logging.info(f"                  = {num_operations/1e+09}GOPs / {total_energy}nJ")
    logging.info(f"                  = {num_operations/total_energy} GOPs/s/W")

    throughput = num_operations/(mywork.total_pipeline_latency * Config.CLK_PERIOD)
    logging.info(f"throughput = num_operations / total latency")
    logging.info(f"                  = {num_operations/1e+09}GOPs / {mywork.total_pipeline_latency * Config.CLK_PERIOD}ns")
    logging.info(f"                  = {throughput} GOPs/s")

    logging.info(f"area efficiency = throughput / area")
    logging.info(f"                  = {throughput}GOPs/s / {total_area/1e+06}mm^2")
    logging.info(f"                  = {throughput/(total_area/1e+06)} GOPs/s/mm^2")


    

    # 計算還沒 parallelize_OU_to_Macro 前的 Macro 數，同時是沒有乘以 K 的，at_least_num_Macro*K 應該要等於 parallelize_OU_to_Macro 內剛開始跑時的 total_macro_num = sum([sum(mywork.PE_num_Macro[i]) * Config.NETWORK_DICT["K"][i]  for i in range(Config.NETWORK_DICT["total_layer_num"])])
    num_OU_without_compression = [ math.ceil(math.ceil(Config.NETWORK_DICT["BIT_W"]/Config.BIT_PER_CELL)*Config.NETWORK_DICT["OUT_CH"][i] / mywork.OU)\
        * math.ceil(Config.NETWORK_DICT["IN_CH"][i] * Config.NETWORK_DICT["K"][i]**2 / mywork.OU) for i in range(Config.NETWORK_DICT["total_layer_num"])]
    logging.info(f"num_OU_without_compression of each layer = {num_OU_without_compression}")
    logging.info(f"sum of num_OU_without_compression of each layer = {sum(num_OU_without_compression)}")


    num_Macro_without_compression = int(math.ceil((sum(num_OU_without_compression)/mywork.num_OU_per_Macro)))
    
    at_least_num_Macro = 0
    for layer_idx in range(Config.NETWORK_DICT["total_layer_num"]):
        for PE_idx in range(Config.NETWORK_DICT["K"][layer_idx]):
            at_least_num_Macro += int(math.ceil(mywork.num_PE_OU_shape[layer_idx][PE_idx] / mywork.num_OU_per_Macro))


    Macro_compression_ratio = at_least_num_Macro / num_Macro_without_compression
    
    logging.info(f"#Macro after compression / #Macro without compression  ")
    logging.info(f"     ={at_least_num_Macro} / {num_Macro_without_compression} ")
    logging.info(f"     ={Macro_compression_ratio}")




    # 計算 unpruned vs pruned+Compression 的 energy 比較
    energy_compression_ratio = [sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])/num_OU_without_compression[i] for i in range(Config.NETWORK_DICT["total_layer_num"])]
    sum_of_unpruned_ADC_energy = sum([mywork.conv_energy_model.ADC_energy[i] / energy_compression_ratio[i] for i in range(Config.NETWORK_DICT["total_layer_num"])])
    sum_of_unpruned_Shift_and_Add_energy = sum([mywork.conv_energy_model.Shift_and_Add_energy[i] / energy_compression_ratio[i] for i in range(Config.NETWORK_DICT["total_layer_num"])])
    logging.info(f"mywork.conv_energy_model.Shift_and_Add_energy = {mywork.conv_energy_model.Shift_and_Add_energy}")
    logging.info(f"energy_compression_ratio = {energy_compression_ratio}")


    logging.info(f"Unpruned")
    logging.info(f"     ADC                                         : {sum_of_unpruned_ADC_energy:>20}nJ")
    logging.info(f"     Shift_and_Add                               : {sum_of_unpruned_Shift_and_Add_energy:>20}nJ")
    logging.info(f"     Overhead(OU_table + Add_and_Distributor)    : {0:>20}nJ")
    logging.info(f"     Else                                        : {(sum_of_Activation_energy + sum_of_Row_Buffers_energy + sum_of_Macro_energy + sum_of_SandH_energy + sum_of_Accumulator_energy + sum(mywork.pooling_energy_Model.Pooling_energy) + mywork.router_energy_model.total_energy_Router):>20}nJ")
    Unpruned_total_energy = sum_of_unpruned_ADC_energy + sum_of_unpruned_Shift_and_Add_energy \
        + (sum_of_Activation_energy + sum_of_Row_Buffers_energy + sum_of_Macro_energy + sum_of_SandH_energy + sum_of_Accumulator_energy + sum(mywork.pooling_energy_Model.Pooling_energy) + mywork.router_energy_model.total_energy_Router)
    logging.info(f"     total                                       : {Unpruned_total_energy:>20}nJ")
    

    logging.info(f"Pruned+Compression")
    logging.info(f"     ADC                                         : {sum_of_ADC_energy:>20}nJ")
    logging.info(f"     Shift_and_Add                               : {sum_of_Shift_and_Add_energy:>20}nJ")
    logging.info(f"     Overhead(OU_table + Add_and_Distributor)    : {(sum_of_which_OU_energy + sum_of_cluster_input_energy + sum_of_weight_bit_position_energy + sum_of_which_filter_energy) + (sum_of_Decoder_energy + sum_of_MUX_of_Shift_and_Add_energy + sum_of_Adder_mask_energy + sum_of_Add_energy + sum_of_MUX_of_filter_energy):>20}nJ")
    logging.info(f"     Else                                        : {(sum_of_Activation_energy + sum_of_Row_Buffers_energy + sum_of_Macro_energy + sum_of_SandH_energy + sum_of_Accumulator_energy + sum(mywork.pooling_energy_Model.Pooling_energy) + mywork.router_energy_model.total_energy_Router):>20}nJ")
    logging.info(f"     total                                       : {total_energy:>20}nJ")


    logging.info(f"save energy                                      :x{total_energy/Unpruned_total_energy:.3f}")
    

    #
    del mywork


if __name__ == '__main__':
    main()
