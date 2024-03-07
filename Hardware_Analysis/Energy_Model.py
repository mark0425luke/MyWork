import Config
import math
import logging
from Hardware_Analysis import Area_Model
logger = logging.getLogger()

class CONV_Energy_Model():
    def __init__(self, mywork):


        # 初始化
        self.Activation_switch_cycle            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Row_Buffers_switch_cycle            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_OU_switch_cycle               = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.cluster_input_switch_cycle          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.weight_bit_position_switch_cycle    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_filter_switch_cycle           = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.OU_switch_cycle                     = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.SandH_switch_cycle                  = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.ADC_switch_cycle                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Shift_and_Add_switch_cycle          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Decoder_switch_cycle                = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.MUX_of_Shift_and_Add_switch_cycle   = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Add_switch_cycle                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Adder_mask_switch_cycle             = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.MUX_of_filter_switch_cycle          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.OR_switch_cycle                     = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Accumulator_switch_cycle            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 


        # 初始化
        self.Activation_energy            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Row_Buffers_energy            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_OU_energy               = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.cluster_input_energy          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.weight_bit_position_energy    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_filter_energy           = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.OU_energy                     = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.SandH_energy                  = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.ADC_energy                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Shift_and_Add_energy          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Decoder_energy                = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.MUX_of_Shift_and_Add_energy   = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Add_energy                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Adder_mask_energy             = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.MUX_of_filter_energy          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.OR_energy                     = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Accumulator_energy            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 


        #
        self.total_energy_CONV               = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]


        # 
        self.calculate_switch_cycle(mywork)
        self.calculate_energy(mywork)
        self.get_energy()

    def calculate_switch_cycle(self, mywork):

        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            
            # ===============================================================
            # Activation
            # ===============================================================
            # 因為是 ex. CONV1 的 output 要給 CONV2 的 Tile 做 Activation
            if i==(Config.NETWORK_DICT["total_layer_num"]-1): # # if last layer, need quant&relu using this layer num_Tile * DUPLICATE
                self.Activation_switch_cycle[i] = mywork.num_Tile[i] * Config.NETWORK_DICT["OUT_CH"][i] * (Config.NETWORK_DICT["OFM_row"][i]**2)
            else: # normal layer, need quant&relu using next CONV layer num_Tile * DUPLICATE
                self.Activation_switch_cycle[i] = mywork.num_Tile[i+1]  * Config.NETWORK_DICT["OUT_CH"][i] * (Config.NETWORK_DICT["OFM_row"][i]**2)
            ###################


            # Row_Buffers
            # self.Row_Buffers_switch_cycle[i]   = mywork.num_Tile[i]  \
            #                                 * math.ceil(Config.NETWORK_DICT["OFM_row"][i] / Config.NETWORK_DICT["K"][i]) * Config.NETWORK_DICT["OFM_row"][i] \
            #                                 * (Config.NETWORK_DICT["CONV_STRIDE"][i] * Config.NETWORK_DICT["K"][i] - Config.NETWORK_DICT["CONV_STRIDE"][i] + Config.NETWORK_DICT["K"][i]) 
            self.Row_Buffers_switch_cycle[i]   = mywork.num_Tile[i]  \
                                            * math.ceil(Config.NETWORK_DICT["OFM_row"][i] / Config.NETWORK_DICT["K"][i]) * Config.NETWORK_DICT["OFM_row"][i]


            # ===============================================================
            # OU_Table
            # ===============================================================
            # common_Macro_dataflow_equation \
            # = math.ceil(Config.NETWORK_DICT["OFM_row"][i]/Config.NETWORK_DICT["K"][i]) * Config.NETWORK_DICT["OFM_row"][i] \
            # * (mywork.Tile_time[i] / mywork.DUPLICATE[i]) # 先不乘 OU
            
            # common_dataflow_equation \
            # = Config.NETWORK_DICT["BLOCK_MIDDLE_TIME"][i] * Config.NETWORK_DICT["OFM_row"][i] \
            # * mywork.Tile_time[i] # 先不乘 OU
            
            how_many_pixel = Config.NETWORK_DICT["BLOCK_MIDDLE_TIME"][i] * Config.NETWORK_DICT["OFM_row"][i]



            # self.which_OU_switch_cycle[i]               = common_dataflow_equation                * mywork.conv_area_model.which_OU_num[i]
            # self.cluster_input_switch_cycle[i]          = common_dataflow_equation                * mywork.conv_area_model.cluster_input_num[i]
            # self.weight_bit_position_switch_cycle[i]    = common_dataflow_equation * mywork.OU    * mywork.conv_area_model.weight_bit_position_num[i]
            # self.which_filter_switch_cycle[i]           = common_dataflow_equation * mywork.OU    * mywork.conv_area_model.which_filter_num[i]

            # 不用乘以 Config.NETWORK_DICT["BIT_IFM"][K]了，因為是共用
            # self.which_OU_switch_cycle[i]               = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel                * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])
            # self.cluster_input_switch_cycle[i]          = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel                * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])
            # self.weight_bit_position_switch_cycle[i]    = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])
            # self.which_filter_switch_cycle[i]           = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])

            # V3
            # self.which_OU_switch_cycle[i]               = how_many_pixel                * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])
            # self.cluster_input_switch_cycle[i]          = how_many_pixel                * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])
            # self.weight_bit_position_switch_cycle[i]    = how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])
            # self.which_filter_switch_cycle[i]           = how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i])



            # V4
            # NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            # self.which_OU_switch_cycle[i]               = 0
            # self.cluster_input_switch_cycle[i]          = how_many_pixel                * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * math.ceil(math.log(NUM_CLUSTER, 2)) / Config.BIT_PER_CELL
            # self.weight_bit_position_switch_cycle[i]    = 0
            # self.which_filter_switch_cycle[i]           = how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * mywork.Tile_MAX_NUM_FILTER[i] \
            #     * math.log(math.ceil(Config.NETWORK_DICT["BIT_W"]/Config.BIT_PER_CELL) * Config.NETWORK_DICT["OUT_CH"][i], 2) / Config.BIT_PER_CELL


            # 改成 BIT_PER_CELL 都用 2 來降低 Macro 數
            NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            self.which_OU_switch_cycle[i]               = 0
            self.cluster_input_switch_cycle[i]          = how_many_pixel                * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * math.ceil(math.log(NUM_CLUSTER, 2)) / 2
            self.weight_bit_position_switch_cycle[i]    = 0
            self.which_filter_switch_cycle[i]           = how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * mywork.Tile_MAX_NUM_FILTER[i] \
                * math.log(math.ceil(Config.NETWORK_DICT["BIT_W"]/2) * Config.NETWORK_DICT["OUT_CH"][i], 2) / 2

            ###################



            # V1
            # self.OU_switch_cycle[i]            = common_dataflow_equation               * mywork.conv_area_model.Macro_num[i]
            # self.SandH_switch_cycle[i]         = common_dataflow_equation               * mywork.conv_area_model.SandH_num[i]
            # self.ADC_switch_cycle[i]           = common_dataflow_equation * mywork.OU   * mywork.conv_area_model.ADC_num[i]
            # self.Shift_and_Add_switch_cycle[i] = common_dataflow_equation * mywork.OU   * mywork.conv_area_model.Shift_and_Add_num[i]
            
            # V2 用 sum_of_PE_num_input_output_for_each_OU_shape 估計
            # common = Config.NETWORK_DICT["BLOCK_MIDDLE_TIME"][i] * Config.NETWORK_DICT["OFM_row"][i] * \
            #     sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i] * Config.NETWORK_DICT["BIT_IFM"]
            # self.OU_switch_cycle[i]            = common               
            # self.SandH_switch_cycle[i]         = common               
            # self.ADC_switch_cycle[i]           = common * mywork.OU   
            # self.Shift_and_Add_switch_cycle[i] = common * mywork.OU   

            
            
            self.OU_switch_cycle[i]            =                                  how_many_pixel                * sum(mywork.sum_of_PE_num_cluster_types_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]                
            self.SandH_switch_cycle[i]         = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_cluster_types_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]                
            self.ADC_switch_cycle[i]           = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_cluster_types_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i] 
            self.Shift_and_Add_switch_cycle[i] = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i] 





            # ===============================================================
            # Add and Distributor
            # ===============================================================
            # V1
            # num_PE = mywork.DUPLICATE[i] * mywork.num_Tile[i] * (Config.NETWORK_DICT["K"][i]**2)
            # self.Decoder_switch_cycle[i]       = common_dataflow_equation   * mywork.OU  * mywork.conv_area_model.Decoder_num[i]
            # # self.MUX_switch_cycle[i]           = common_dataflow_equation   * mywork.OU  * mywork.conv_area_model.MUX_num[i]
            # self.MUX_switch_cycle[i]           = 0
            # self.Add_switch_cycle[i]           = common_dataflow_equation   * mywork.OU  * num_PE * Config.num_Macro_per_Tile


            '''
            # V2 用 sum_of_PE_num_input_output_for_each_OU_shape 估計
            # self.Decoder_switch_cycle[i]       = common * mywork.OU 
            # # self.MUX_switch_cycle[i]           = common_dataflow_equation   * mywork.OU  * mywork.conv_area_model.MUX_num[i]
            # self.MUX_switch_cycle[i]           = 0
            # self.Add_switch_cycle[i]           = common * mywork.OU 

            self.Decoder_switch_cycle[i]       = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]  
            self.MUX_switch_cycle[i]           = 0
            self.Add_switch_cycle[i]           = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU    * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i] * Config.NETWORK_DICT["K"][i]
            '''
            
            # V3 Adder_mask
            # self.Decoder_switch_cycle[i]        = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
            # self.MUX_switch_cycle[i]            = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
            # self.Adder_mask_switch_cycle[i]     = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
            # self.Add_switch_cycle[i]            = Config.NETWORK_DICT["BIT_IFM"] * how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]


            self.Decoder_switch_cycle[i]                =   how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
            self.MUX_of_Shift_and_Add_switch_cycle[i]   =   how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
            self.Adder_mask_switch_cycle[i]             =   how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
            self.Add_switch_cycle[i]                    =   how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
             # MUX_of_filter 拿掉了，因為改成 OR
            # self.MUX_of_filter_switch_cycle[i]          =   how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
            self.MUX_of_filter_switch_cycle[i]          = 0
            self.OR_switch_cycle[i]                      = how_many_pixel * mywork.OU * sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * Config.NETWORK_DICT["K"][i]
                       
            ###################

            
           

            # self.Accumulator_switch_cycle[i] = common_dataflow_equation * (Config.NETWORK_DICT["K"][i]) * Config.NETWORK_DICT["OUT_CH"][i]
            self.Accumulator_switch_cycle[i] = how_many_pixel * mywork.conv_area_model.Accumulator_num[i]


    def calculate_energy(self, mywork):
        
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
        
            # V1 正確版
            # self.Activation_energy[i]              = self.Activation_switch_cycle[i]          * Config.CLK_PERIOD * Config.Activation[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], Config.NETWORK_DICT["BIT_OFM"])]["power"]
            
            # V2, 改用 OFM 29-bit
            self.Activation_energy[i]              = self.Activation_switch_cycle[i]          * Config.CLK_PERIOD * Config.Activation[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], 29)]["power"]
            
            
            self.Row_Buffers_energy[i]             = self.Row_Buffers_switch_cycle[i]         * Config.CLK_PERIOD * Config.Row_Buffers[(Config.CLK_PERIOD, Config.NETWORK_DICT["SRAM_NUM_WORD"][i], Config.NETWORK_DICT["SRAM_WIDTH"][i])]["power"]
            
            
            # OU_Table, 從 Mux_base 挑 

            # V1 正確版
            # self.which_OU_energy[i]                = self.which_OU_switch_cycle[i]            * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, math.log(mywork.num_OU_per_Macro, 2))]["power"] 
            # self.cluster_input_energy[i]           = self.cluster_input_switch_cycle[i]       * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, mywork.OU)]["power"]  
            # self.weight_bit_position_energy[i]     = self.weight_bit_position_switch_cycle[i] * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, Config.NETWORK_DICT["BIT_W"])]["power"]     
            # self.which_filter_energy[i]            = self.which_filter_switch_cycle[i]        * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, math.log(Config.NETWORK_DICT["OUT_CH"][i], 2))]["power"]      
            
            
            # V2, 改都用最大的 BIT=32 
            # self.which_OU_energy[i]                = self.which_OU_switch_cycle[i]            * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"] 
            # self.cluster_input_energy[i]           = self.cluster_input_switch_cycle[i]       * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]  
            # self.weight_bit_position_energy[i]     = self.weight_bit_position_switch_cycle[i] * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]     
            # self.which_filter_energy[i]            = self.which_filter_switch_cycle[i]        * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]      
            
            
            
            # V3, 改用 Macro, MUX 跟 Macro 的轉換是  MUX_bit_數 * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(1, 1.28)]["power"] * 0.78125), 0.78125ns = 1/1.28GHz   
            # self.which_OU_energy[i]              = self.which_OU_switch_cycle[i]            * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(1, 1.28)]["power"] * 0.78125)
            # self.cluster_input_energy[i]         = self.cluster_input_switch_cycle[i]       * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(1, 1.28)]["power"] * 0.78125)
            # self.weight_bit_position_energy[i]   = self.weight_bit_position_switch_cycle[i] * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(1, 1.28)]["power"] * 0.78125)
            # self.which_filter_energy[i]          = self.which_filter_switch_cycle[i]        * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(1, 1.28)]["power"] * 0.78125)


            # V4
            # self.which_OU_energy[i]              = 0
            # self.cluster_input_energy[i]         = self.cluster_input_switch_cycle[i]       * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(Config.BIT_PER_CELL, 1.28)]["power"] * Config.CLK_PERIOD)
            # self.weight_bit_position_energy[i]   = 0
            # self.which_filter_energy[i]          = self.which_filter_switch_cycle[i]        * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(Config.BIT_PER_CELL, 1.28)]["power"] * Config.CLK_PERIOD)


            # 改成 BIT_PER_CELL 都用 2 來降低 Macro 數
            self.which_OU_energy[i]              = 0
            self.cluster_input_energy[i]         = self.cluster_input_switch_cycle[i]       * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(2, 1.28)]["power"] * Config.CLK_PERIOD)
            self.weight_bit_position_energy[i]   = 0
            self.which_filter_energy[i]          = self.which_filter_switch_cycle[i]        * (Config.Macro[(0)]["device_power"] * Config.Macro[(0)]["crossbar_read_latency"] + Config.ADC[(2, 1.28)]["power"] * Config.CLK_PERIOD)

            ###################
            


            self.OU_energy[i]                      = self.OU_switch_cycle[i]   *   Config.Macro[(0)]["device_power"]   *   Config.Macro[(0)]["crossbar_read_latency"]   *   mywork.OU   *   mywork.OU
            self.SandH_energy[i]                   = self.SandH_switch_cycle[i]               * Config.CLK_PERIOD * Config.SandH[(0)]["power"]
            self.ADC_energy[i]                     = self.ADC_switch_cycle[i]                 * Config.CLK_PERIOD * Config.ADC[(mywork.ADC_PRECISION, Config.ADC_GSps)]["power"]  


            self.Shift_and_Add_energy[i]           = self.Shift_and_Add_switch_cycle[i]       * Config.CLK_PERIOD * Config.Shift_and_Add[(Config.CLK_PERIOD, Config.BIT_PER_CELL, Config.BIT_DAC, mywork.OU, Config.NETWORK_DICT["BIT_W"],  Config.NETWORK_DICT["BIT_IFM"])]["power"]


            # Add and Distributor
            


            # V1 正確版
            # BIT_INPUT_SHIFT = mywork.ADC_PRECISION  + Config.NETWORK_DICT["BIT_IFM"] + Config.NETWORK_DICT["W"] -1
            # self.Decoder_energy[i]                 = self.Decoder_switch_cycle[i]             * Config.CLK_PERIOD * Config.Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["power"]
            # self.MUX_energy[i]                     = self.MUX_switch_cycle[i]                 * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, BIT_INPUT_SHIFT)]["power"]
            # self.Add_energy[i]                   = self.Add_switch_cycle[i]               * Config.CLK_PERIOD * self.Add_num[i] * Config.Add[(Config.CLK_PERIOD, \
            #                             Config.NETWORK_DICT["BIT_IFM"] + Config.NETWORK_DICT["W"] + math.ceil(math.log(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i], 2)))]["power"]

           
#            
#             # V2 改成固定用 MUX BIT=32, Add BIT=29
#             BIT_INPUT_SHIFT = mywork.ADC_PRECISION  + Config.NETWORK_DICT["BIT_IFM"] + Config.NETWORK_DICT["BIT_W"] -1
#             
#             self.Decoder_energy[i]                 = self.Decoder_switch_cycle[i]             * Config.CLK_PERIOD * Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["power"]
#             # self.Decoder_energy[i]                 = 0
# 
#             # self.MUX_energy[i]                     = self.MUX_switch_cycle[i]                 * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["power"]
#             self.MUX_energy[i]                     = 0
# 
#             self.Add_energy[i]                      = self.Add_switch_cycle[i] * Config.CLK_PERIOD * Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["power"]
#             # self.Add_energy[i]                      = self.Add_switch_cycle[i] * Config.CLK_PERIOD * Config.Add_base[(Config.CLK_PERIOD, 3, 28, 29)]["power"]
#             # self.Add_energy[i]                      = 0
            
            
            
            # V3 Adder_mask
            self.Decoder_energy[i]                  = self.Decoder_switch_cycle[i]             * Config.CLK_PERIOD * Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["power"]
            
            self.MUX_of_Shift_and_Add_energy[i]     = self.MUX_of_Shift_and_Add_switch_cycle[i] * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], 21)]["power"]
            
            # Adder 理論上是隨著 Macro數/BIT_FIM 而變得，但是我懶得改了
            self.Adder_mask_energy[i]               = self.Adder_mask_switch_cycle[i]          * Config.CLK_PERIOD * Config.Adder_mask[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, Config.NETWORK_DICT["OUT_CH"][i])]["power"]
            
            # ex. PE0 16 Macro, PE1 32 Macro, PE2 48 Macro
            # 除以 BIT_IFM 後
            # 得到 PE0 需要 2 個 2-input Add
            # 得到 PE1 需要 4 個 2-input Add
            # 得到 PE2 需要 6 個 2-input Add
            how_many_pixel = Config.NETWORK_DICT["BLOCK_MIDDLE_TIME"][i] * Config.NETWORK_DICT["OFM_row"][i]
            self.Add_energy[i] = sum([\
                    mywork.sum_of_PE_num_input_output_for_each_OU_shape[i][j]  \
                    * Area_Model.Interpolate(\
                        upper_level=Config.MAX_num_Macro_per_Tile, \
                        lower_level=2, \
                        your_level=math.ceil(mywork.PE_num_Macro[i][j]/Config.NETWORK_DICT["BIT_IFM"]), \
                        upper_value=Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["power"], \
                        lower_value=Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["power"]) for j in range(Config.NETWORK_DICT["K"][i])])\
                    * Config.NETWORK_DICT["K"][i] * how_many_pixel * mywork.OU
            
            
            self.MUX_of_filter_energy[i]            = self.MUX_of_filter_switch_cycle[i]                 * Config.CLK_PERIOD * Config.Mux_base[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i],\
                math.ceil(math.log(Config.NETWORK_DICT["IN_CH"][i]*(Config.NETWORK_DICT["K"][i]**2),2))+Config.NETWORK_DICT["BIT_IFM"]+Config.NETWORK_DICT["BIT_W"]-1)]["power"]
            
            self.OR_energy[i]                       = self.OR_switch_cycle[i]                           * Config.CLK_PERIOD * Config.OR_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 29)]["power"]
            ###################

            



            self.Accumulator_energy[i]             = self.Accumulator_switch_cycle[i]         * Config.CLK_PERIOD * Config.Add_base[(Config.CLK_PERIOD, Config.NETWORK_DICT["K"][i], 28, 29)]["power"]


            # total energy
            self.total_energy_CONV[i] \
                = self.Activation_energy[i] + self.Row_Buffers_energy[i] \
                + self.which_OU_energy[i] + self.cluster_input_energy[i] + self.weight_bit_position_energy[i] + self.which_filter_energy[i] \
                + self.OU_energy[i] \
                + self.SandH_energy[i] + self.ADC_energy[i] + self.Shift_and_Add_energy[i]\
                + self.Decoder_energy[i] + self.MUX_of_Shift_and_Add_energy[i] +self.Add_energy[i] + self.Adder_mask_energy[i] + self.MUX_of_filter_energy[i] + self.OR_energy[i]\
                + self.Accumulator_energy[i]

            ###################


    def get_energy(self):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):

            logger.info(f" CONV{i+1}: ")
            logger.info(f"     Activation             switch_cycle = {self.Activation_switch_cycle[i]:>20}nJ")
            logger.info(f"                            energy       = {self.Activation_energy[i]:>20}nJ")


            logger.info(f"     Row_Buffers            switch_cycle = {self.Row_Buffers_switch_cycle[i]:>20}nJ")
            logger.info(f"                            energy       = {self.Row_Buffers_energy[i]:>20}nJ")

            logger.info(f"     OU_Table              ")
            logger.info(f"            which_OU            switch_cycle    = {self.which_OU_switch_cycle[i]:>20}")
            logger.info(f"                                energy          = {self.which_OU_energy[i]:>20}")
            logger.info(f"            cluster_input       switch_cycle     = {self.cluster_input_switch_cycle[i]:>20}")
            logger.info(f"                                energy           = {self.cluster_input_energy[i]:>20}")
            logger.info(f"            weight_bit_position switch_cycle    = {self.weight_bit_position_switch_cycle[i]:>20}")
            logger.info(f"                                energy          = {self.weight_bit_position_energy[i]:>20}")
            logger.info(f"            which_filter        switch_cycle    = {self.which_filter_switch_cycle[i]:>20}")
            logger.info(f"                                energy          = {self.which_filter_energy[i]:>20}")
            logger.info(f"            OU_Table energy            = {self.which_OU_energy[i] + self.cluster_input_energy[i] + self.weight_bit_position_energy[i] + self.which_filter_energy[i]:>20}nJ")

            logger.info(f"     OU                        switch_cycle = {self.OU_switch_cycle[i]:>20}nJ") 
            logger.info(f"                                energy       = {self.OU_energy[i]:>20}nJ") 
            
            logger.info(f"     SandH                      switch_cycle = {self.SandH_switch_cycle[i]:>20}nJ") 
            logger.info(f"                                energy       = {self.SandH_energy[i]:>20}nJ") 
    
            logger.info(f"     ADC                      switch_cycle = {self.ADC_switch_cycle[i]:>20}nJ") 
            logger.info(f"                                energy       = {self.ADC_energy[i]:>20}nJ") 

            logger.info(f"     Shift_and_Add              switch_cycle = {self.Shift_and_Add_switch_cycle[i]:>20}nJ") 
            logger.info(f"                                energy       = {self.Shift_and_Add_energy[i]:>20}nJ") 

            logger.info(f"     Add and Distributor ")
            logger.info(f"            Decoder                   switch_cycle    = {self.Decoder_switch_cycle[i]:>20}")
            logger.info(f"                                      energy          = {self.Decoder_energy[i]:>20}")
            logger.info(f"            MUX_of_Shift_and_Add      switch_cycle = {self.MUX_of_Shift_and_Add_switch_cycle[i]:>20}")
            logger.info(f"                                      energy = {self.MUX_of_Shift_and_Add_energy[i]:>20}um^2")
            logger.info(f"            Adder_mask                switch_cycle    = {self.Adder_mask_switch_cycle[i]:>20}")
            logger.info(f"                                      energy          = {self.Adder_mask_energy[i]:>20}")
            logger.info(f"            Add                       switch_cycle    = {self.Add_switch_cycle[i]:>20}")
            logger.info(f"                                      energy          = {self.Add_energy[i]:>20}")
            logger.info(f"           MUX_of_filter              switch_cycle = {self.MUX_of_filter_switch_cycle[i]:>20}")
            logger.info(f"                                      energy = {self.MUX_of_filter_energy[i]:>20}")
            logger.info(f"           OR                         switch_cycle = {self.OR_switch_cycle[i]:>20}")
            logger.info(f"                                      energy = {self.OR_energy[i]:>20}")
            logger.info(f"            Add and Distributor energy = {self.Decoder_energy[i] + self.MUX_of_Shift_and_Add_energy[i] + self.Adder_mask_energy[i] +  +self.Add_energy[i] + self.MUX_of_filter_energy[i] + self.OR_energy[i]:>20}nJ")
       
            

            logger.info(f"    Accumulator             switch_cycle    = {self.Accumulator_switch_cycle[i]:>20}")
            logger.info(f"                            energy          = {self.Accumulator_energy[i]:>20}")
            
            logger.info(f"     total_energy               = {self.total_energy_CONV[i]:>20}nJ")

class Pooling_Energy_Model():
    def __init__(self, mywork):
        # 初始化硬體個數的參數
        self.Pooling_switch_cycle        = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
       

        # 初始化硬體area的參數
        self.Pooling_energy              = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
     

        #
        self.calculate_switch_cycle(mywork)
        self.calculate_energy()
        self.get_energy()

    def calculate_switch_cycle(self, mywork):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            if Config.NETWORK_DICT["POOLING_SIZE"][i]!=0 : # 代表這層 CONV 後面才有接 Pooling, 其中一般CONVi 的 output 是 CONVi+1 負責 Pool，最後CONV 的 Pooling 還是由最後 Pool
                
                if(i==Config.NETWORK_DICT["total_layer_num"]-1): # 最後一層
                    # self.Pooling_switch_cycle[i] = mywork.num_Tile[i] * mywork.DUPLICATE[i] \
                    # * (((Config.NETWORK_DICT["OFM_row"][i] - Config.NETWORK_DICT["POOLING_SIZE"][i] ) / Config.NETWORK_DICT["POOLING_STRIDE"][i] )+1)**2  * Config.NETWORK_DICT["OUT_CH"][i]

                    self.Pooling_switch_cycle[i] = mywork.num_Tile[i]  \
                    * (((Config.NETWORK_DICT["OFM_row"][i] - Config.NETWORK_DICT["POOLING_SIZE"][i] ) / Config.NETWORK_DICT["POOLING_STRIDE"][i] )+1)**2  * Config.NETWORK_DICT["OUT_CH"][i]
            
                else:
                    # self.Pooling_switch_cycle[i] = mywork.num_Tile[i+1] * mywork.DUPLICATE[i+1] \
                    # * (((Config.NETWORK_DICT["OFM_row"][i] - Config.NETWORK_DICT["POOLING_SIZE"][i] ) / Config.NETWORK_DICT["POOLING_STRIDE"][i] )+1)**2  * Config.NETWORK_DICT["OUT_CH"][i]

                    self.Pooling_switch_cycle[i] = mywork.num_Tile[i+1] \
                    * (((Config.NETWORK_DICT["OFM_row"][i] - Config.NETWORK_DICT["POOLING_SIZE"][i] ) / Config.NETWORK_DICT["POOLING_STRIDE"][i] )+1)**2  * Config.NETWORK_DICT["OUT_CH"][i]
            
            else:
                self.Pooling_switch_cycle[i] = 0

    def calculate_energy(self):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            if Config.NETWORK_DICT["POOLING_SIZE"][i]!=0 : # 代表這層 CONV 後面才有接 Pooling, 其中一般CONVi 的 output 是 CONVi+1 負責 Pool，最後CONV 的 Pooling 還是由最後 Pool
                self.Pooling_energy[i]                 = self.Pooling_switch_cycle[i]             * Config.CLK_PERIOD * Config.Pooling[(Config.CLK_PERIOD, Config.NETWORK_DICT["POOLING_SIZE"][i], Config.NETWORK_DICT["BIT_IFM"])]["power"]
            else :
                self.Pooling_energy[i] = 0

    def get_energy(self):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            print(f" CONV{i+1}: ")
            print(f"    Pooling energy = {self.Pooling_energy[i]:>20}um^2")

class Router_Energy_Model():
    def __init__(self, mywork):

        self.Router_switch_cycle            = 0
        # self.Router_energy = 0
        self.total_energy_Router             = 0


        #
        self.calculate_switch_cycle(mywork)
        self.calculate_energy()
        self.get_energy()


    def calculate_switch_cycle(self, mywork):
        
        for i in range(Config.NETWORK_DICT["total_layer_num"] - 1):
            curConv_inner_switch_cycle            = Config.NETWORK_DICT["OUT_CH"][i] * (Config.NETWORK_DICT["OFM_row"][i]**2) * mywork.router_area_model.intra_CONV_Router_num[i]
            curConv_to_nextCONV_switch_cycle       = 1 # 假設 1 就好，實際跑 QAP 也是 router energy 很小，更何況 Tile 個數不多 
            nextCONV_inner_switch_cycle          = Config.NETWORK_DICT["OUT_CH"][i+1] * (Config.NETWORK_DICT["OFM_row"][i+1]**2) * mywork.router_area_model.intra_CONV_Router_num[i+1]

            self.Router_switch_cycle += (   curConv_inner_switch_cycle \
                                            + curConv_to_nextCONV_switch_cycle \
                                            + nextCONV_inner_switch_cycle)


    def calculate_energy(self):
        self.total_energy_Router = self.Router_switch_cycle   * Config.CLK_PERIOD * Config.Router[(Config.CLK_PERIOD, 29)]["power"]

    def get_energy(self):
        print(f" total Router energy = {self.total_energy_Router:>20}um^2")