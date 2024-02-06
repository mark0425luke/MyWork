import Config
import math
import logging
logger = logging.getLogger()

def Decomposition(Large_input, Small_input):
    tmp = Large_input
    quotient = 0
    while( not tmp==1):
        tmp = math.ceil(tmp/Small_input)
        quotient += tmp

    return quotient        

def Interpolate(upper_level, lower_level, your_level, upper_value, lower_value):
    your_value = lower_value + (upper_value-lower_value) * (your_level-lower_level) / (upper_level-lower_level)
    return your_value

class CONV_Area_Model():
    def __init__(self, mywork):

        # 初始化硬體個數的參數
        self.Activation_num             = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Row_Buffers_num            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_OU_num               = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.cluster_input_num          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.weight_bit_position_num    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_filter_num           = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Macro_num                  = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.SandH_num                  = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.ADC_num                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Shift_and_Add_num          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Decoder_num                = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.MUX_of_Shift_and_Add_num   = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Add_num                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Adder_mask_num             = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.MUX_of_filter_num          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Accumulator_num            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 


        # 初始化硬體area的參數
        self.Activation_area             = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Row_Buffers_area            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_OU_area               = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.cluster_input_area          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.weight_bit_position_area    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.which_filter_area           = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Macro_area                  = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.SandH_area                  = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.ADC_area                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Shift_and_Add_area          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.Decoder_area                = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.MUX_of_Shift_and_Add_area   = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Add_area                    = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Adder_mask_area             = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.MUX_of_filter_area          = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 
        self.Accumulator_area            = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])] 


        
        
        
        # 初始化 CONV 總面積的參數
        self.total_area_CONV        = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        

        # need to run first
        self.calculate_num(mywork)
        
        # based on calculate_num, calculate area
        self.calculate_area(mywork)


        # print the answer
        self.get_area()




    def calculate_num(self, mywork):

        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            # Activation, 因為 CONV1 的 Activation 是做在 CONV2
            # 所以是拿 (上層 NUM_DATA/ 上層 NUM_CYCLE) * (這層 num_Tile * 這層 DUPLICATE)
            # 例外是最後CONV
            NUM_DATA    = Config.NETWORK_DICT["OUT_CH"][i] * Config.NETWORK_DICT["K"][i]
            NUM_CYCLE   = math.floor( mywork.BOTTLENECK_LATENCY / Config.NETWORK_DICT["DELTA_X"][i]) # 不用乘以 OU, 因為 BOTTLENECK_LATENCY = BOTTLENECK_OU_CYCLE * OU
            
            if i==(Config.NETWORK_DICT["total_layer_num"]-1): # 最後一層 CONV
                self.Activation_num[i] = mywork.num_Tile[i] * math.ceil(NUM_DATA / NUM_CYCLE)
            else:
                self.Activation_num[i] = mywork.num_Tile[i+1] * math.ceil(NUM_DATA / NUM_CYCLE)
            
            
            
            # Row_Buffers, verilog 已經是 S*K - S + K 個了
            self.Row_Buffers_num[i] = mywork.num_Tile[i] 
            

            # ===============================================================
            # OU_Table
            # ===============================================================
            NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            num_Macro_shared_for_PE = sum(mywork.PE_num_Macro[i])


            # # V1, 用 mux 去兜
            # self.which_OU_num[i]               =  num_Macro_shared_for_PE \
            #     * Decomposition(Large_input=(mywork.Tile_time[i]/mywork.DUPLICATE[i]), Small_input=Config.Mux_base_Small_input)
            # self.cluster_input_num[i]          = num_Macro_shared_for_PE \
            #     * Decomposition(Large_input=(Config.NETWORK_DICT["BIT_IFM"] * NUM_CLUSTER), Small_input=Config.Mux_base_Small_input)
            # self.weight_bit_position_num[i]    = num_Macro_shared_for_PE *  mywork.Tile_MAX_NUM_FILTER[i]  \
            #     * Decomposition(Large_input=(mywork.Tile_time[i]/mywork.DUPLICATE[i]), Small_input=Config.Mux_base_Small_input)
            # self.which_filter_num[i]           = num_Macro_shared_for_PE *  mywork.Tile_MAX_NUM_FILTER[i] \
            #     * Decomposition(Large_input=(mywork.Tile_time[i]/mywork.DUPLICATE[i]), Small_input=Config.Mux_base_Small_input)


            # V3, 用 Macro, 這邊先求 #Macro 或是 #ADC
            # which_OU 可以拿掉，因為只需要照著順序擺 OU，然後做個前後吃不同 cluster_input 的判斷電路就好
            # 現在沒有 MAX_NUM_FILTER 了，所以一條 bitline 只會有一個 filter，那就代表只需要像是 RePIM 一樣紀錄
            # self.which_OU_num[i]               = num_Macro_shared_for_PE * math.log(mywork.num_OU_per_Macro, 2) 
            # self.cluster_input_num[i]          = num_Macro_shared_for_PE * mywork.OU 
            # self.weight_bit_position_num[i]    = num_Macro_shared_for_PE *  mywork.Tile_MAX_NUM_FILTER[i]  * Config.NETWORK_DICT["BIT_W"] 
            # self.which_filter_num[i]           = num_Macro_shared_for_PE *  mywork.Tile_MAX_NUM_FILTER[i] * math.log(Config.NETWORK_DICT["OUT_CH"][i], 2) 


            # V4
            # which_OU 可以拿掉，因為只需要照著順序擺 OU，然後做個前後吃不同 cluster_input 的判斷電路就好
            # 現在沒有 MAX_NUM_FILTER 了，所以一條 bitline 只會有一個 filter，
            # 那就代表只需要像是 RePIM 一樣紀錄原本是哪一條 column 就好
            # 哪個 filter 跟 bit position 可以還原
            # Filter number = ⌊col idx / QW C ⌋, and
            # Bit position = (col idx mod QW C ) · bit-per-cell, where
            # QW C = ⌈QW / bit-per-cell⌉ and bit-per-cell corresponds to
            # the bit resolution of a ReRAM cell. Therefore, storing col idx
            # is enough to represent both filter number and bit position of
            # the orignal DNN weight matrix.
            
            
            # NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            # self.which_OU_num[i]               = 0
            # self.cluster_input_num[i]          = num_Macro_shared_for_PE * math.ceil(math.log(NUM_CLUSTER, 2)) / Config.BIT_PER_CELL / mywork.OU
            # self.weight_bit_position_num[i]    = 0
            # self.which_filter_num[i]           = num_Macro_shared_for_PE *  mywork.Tile_MAX_NUM_FILTER[i] \
            #     * math.log(math.ceil(Config.NETWORK_DICT["BIT_W"]/Config.BIT_PER_CELL) * Config.NETWORK_DICT["OUT_CH"][i], 2) / Config.BIT_PER_CELL / mywork.OU
            

            
            # 改成 BIT_PER_CELL 都用 2 來降低 Macro 數
            # NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            # self.which_OU_num[i]               = 0
            # self.cluster_input_num[i]          = num_Macro_shared_for_PE * math.ceil(math.log(NUM_CLUSTER, 2)) / 2 / mywork.OU
            # self.weight_bit_position_num[i]    = 0
            # self.which_filter_num[i]           = num_Macro_shared_for_PE *  mywork.Tile_MAX_NUM_FILTER[i] \
            #     * math.log(math.ceil(Config.NETWORK_DICT["BIT_W"]/2) * Config.NETWORK_DICT["OUT_CH"][i], 2) / 2 / mywork.OU


            
            # 改成等效上總共多少 cell / (128*128) 得到 #Macro, 且 BIT_PER_CELL 都用 2
            NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            self.which_OU_num[i]               = 0
            self.cluster_input_num[i]          = sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * math.ceil(math.log(NUM_CLUSTER, 2)) / 2 / (128**2)
            self.weight_bit_position_num[i]    = 0
            self.which_filter_num[i]           = sum(mywork.sum_of_PE_num_input_output_for_each_OU_shape[i]) * math.log(math.ceil(Config.NETWORK_DICT["BIT_W"]/2) * Config.NETWORK_DICT["OUT_CH"][i], 2) / 2 / (128**2)
            ###################




            # 每個 Macro 都有的 SandH, ADC, Shift_and_Add(Shift_and_Add 已經改成只有一個Calculate.v 情況了)
            num_Macro = sum(mywork.PE_num_Macro[i]) * Config.NETWORK_DICT["K"][i]
            self.Macro_num[i] = num_Macro
            self.SandH_num[i] = num_Macro * Config.XB_paramX
            self.ADC_num[i] = num_Macro 
            self.Shift_and_Add_num[i] = num_Macro * mywork.Tile_MAX_NUM_FILTER[i]
            


            # ===============================================================
            # Add_and_Distributor
            # ===============================================================
            
            '''
            # V1 
            self.Decoder_num[i] = num_Macro
            # self.MUX_num[i]     = num_PE * Config.NETWORK_DICT["OUT_CH"][i] * Config.num_Macro_per_Tile\
            #     * Decomposition(Large_input=mywork.Tile_MAX_NUM_FILTER[i], Small_input=Config.Mux_base_Small_input)
            self.MUX_num[i]     = 0

            
            # num_PE = mywork.DUPLICATE[i] * mywork.num_Tile[i] * (Config.NETWORK_DICT["K"][i]**2)
            # self.Add_num[i] = num_PE * Config.NETWORK_DICT["OUT_CH"][i] * \
            #      Decomposition(Large_input=Config.num_Macro_per_Tile, Small_input=Config.Add_and_Distributor_Small_input)
            

            
            # 譬如 PE0 9 Macro, PE1 17 Macro, PE2 20 Macro, 所以需要開 2 個 Tile, 
            # PE0 需要 OUT_CH   * K 個 16-input Add
            # PE1 需要 2*OUT_CH * K 個 16-input Add
            # PE2 需要 2*OUT_CH * K 個 16-input Add
            self.Add_num[i] = Config.NETWORK_DICT["OUT_CH"][i] * \
                sum([ math.ceil(mywork.PE_num_Macro[i][j] / Config.MAX_num_Macro_per_Tile) for j in range(Config.NETWORK_DICT["K"][i])]) * Config.NETWORK_DICT["K"][i] 
            '''

            # V3  Adder_mask, 除以 BIT_IFM 是因為每 BIT_IFM 個 cycle 才需要 switch 一次
            self.Decoder_num[i]                 = sum([math.ceil(mywork.PE_num_Macro[i][j]/Config.NETWORK_DICT["BIT_IFM"]) for j in range(Config.NETWORK_DICT["K"][i])]) * Config.NETWORK_DICT["K"][i]
            self.MUX_of_Shift_and_Add_num[i]    = sum([math.ceil(mywork.PE_num_Macro[i][j]/Config.NETWORK_DICT["BIT_IFM"]) for j in range(Config.NETWORK_DICT["K"][i])]) * Config.NETWORK_DICT["K"][i]
            self.Adder_mask_num[i]              = mywork.num_Tile[i] * (Config.NETWORK_DICT["K"][i]**2)
            self.Add_num[i]                     = sum([math.ceil(mywork.PE_num_Macro[i][j]/Config.NETWORK_DICT["BIT_IFM"]) for j in range(Config.NETWORK_DICT["K"][i])]) * Config.NETWORK_DICT["K"][i]
            self.MUX_of_filter_num[i]           = sum([math.ceil(mywork.PE_num_Macro[i][j]/Config.NETWORK_DICT["BIT_IFM"]) for j in range(Config.NETWORK_DICT["K"][i])]) * Config.NETWORK_DICT["K"][i]
            
            ###################


            self.Accumulator_num[i] = mywork.num_Tile[i] *  Config.NETWORK_DICT["K"][i] * Config.NETWORK_DICT["OUT_CH"][i]



    def calculate_area(self, mywork):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            # V1 正確版
            # self.Activation_area[i] = self.Activation_num[i] * Config.Activation[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], Config.NETWORK_DICT["BIT_OFM"])]["area"]
            
            # V2, 改用 OFM 29-bit
            self.Activation_area[i] = self.Activation_num[i] * Config.Activation[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], 29)]["area"]



            self.Row_Buffers_area[i] = self.Row_Buffers_num[i] * Config.Row_Buffers[(Config.CLK_PERIOD, Config.NETWORK_DICT["SRAM_NUM_WORD"][i], Config.NETWORK_DICT["SRAM_WIDTH"][i])]["area"]
            

            # ===============================================================
            # OU_Table, 從 Mux_base 挑, 現在 MUX 改成 Macro 
            # ===============================================================
            NUM_OU_PER_MACRO = (Config.XB_paramX / mywork.OU)**2

            # V1 正確版
            # self.which_OU_area[i] = self.which_OU_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, math.log(mywork.num_OU_per_Macro, 2))]["area"]                
            # self.cluster_input_area[i] = self.cluster_input_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, mywork.OU)]["area"]            
            # self.weight_bit_position_area[i] = self.weight_bit_position_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, Config.NETWORK_DICT["BIT_W"])]["area"]      
            # self.which_filter_area[i] = self.which_filter_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, math.log(Config.NETWORK_DICT["OUT_CH"][i], 2))]["area"]  


            # V2, 改都用最大的 BIT=32 
            # self.which_OU_area[i] = self.which_OU_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input,32)]["area"]                
            # self.cluster_input_area[i] = self.cluster_input_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["area"]            
            # self.weight_bit_position_area[i] = self.weight_bit_position_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["area"]      
            # self.which_filter_area[i] = self.which_filter_num[i] * \
            #     Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["area"]        



            # V3, 改用 Macro, MUX 跟 Macro 的轉換是  MUX_bit_數 * ( (input個數) *Config.Macro[(0)]["device_area"] + Config.ADC[ (1, 1.28)]["area"] )     
            # NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            # self.which_OU_area[i]             = self.which_OU_num[i] \
            #     * ((mywork.Tile_time[i]/mywork.DUPLICATE[i]) * Config.Macro[(0)]["device_area"]         + Config.ADC[ (1, 1.28)]["area"] )
            # self.cluster_input_area[i]        = self.cluster_input_num[i] \
            #     * ( (Config.NETWORK_DICT["BIT_IFM"] * NUM_CLUSTER) * Config.Macro[(0)]["device_area"]    + Config.ADC[ (1, 1.28)]["area"] )     
            # self.weight_bit_position_area[i]  = self.weight_bit_position_num[i] \
            #     *  ( (mywork.Tile_time[i]/mywork.DUPLICATE[i]) * Config.Macro[(0)]["device_area"]         + Config.ADC[ (1, 1.28)]["area"] )
            # self.which_filter_area[i]         = self.which_filter_num[i] \
            #     * ( (mywork.Tile_time[i]/mywork.DUPLICATE[i]) * Config.Macro[(0)]["device_area"]         + Config.ADC[ (1, 1.28)]["area"] )     

            
            # V4
            # 幾個 cell 估計版本
#             self.which_OU_area[i]             = 0
#             NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
#             cluster_input_ADC_area = self.cluster_input_num[i] * Config.ADC[ (Config.BIT_PER_CELL, 1.28)]["area"]
#             cluster_input_Macro_area = sum([  mywork.sum_of_PE_num_input_output_for_each_OU_shape[i][j] *  math.ceil(math.log(NUM_CLUSTER, 2)) / Config.BIT_PER_CELL * Config.Macro[(0)]["device_area"] for j in range(Config.NETWORK_DICT["K"][i])])
#             self.cluster_input_area[i]        = cluster_input_ADC_area + cluster_input_Macro_area
#     
#             self.weight_bit_position_area[i]  = 0
# 
#             which_filter_ADC_area = self.which_filter_num[i] * Config.ADC[ (Config.BIT_PER_CELL, 1.28)]["area"]
#             which_filter_Macro_area = sum([ mywork.sum_of_PE_num_input_output_for_each_OU_shape[i][j] * math.log(math.ceil(Config.NETWORK_DICT["BIT_W"]/Config.BIT_PER_CELL) * Config.NETWORK_DICT["OUT_CH"][i], 2) / Config.BIT_PER_CELL * Config.Macro[(0)]["device_area"] for j in range(Config.NETWORK_DICT["K"][i]) ])
#             self.which_filter_area[i]         = which_filter_ADC_area + which_filter_Macro_area



            # 每個 Macro 都 128x128 版本
#             self.which_OU_area[i]             = 0
#             NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
#             cluster_input_ADC_area = self.cluster_input_num[i] * Config.ADC[ (Config.BIT_PER_CELL, 1.28)]["area"]
#             cluster_input_Macro_area = self.cluster_input_num[i] * 128*128 *  Config.Macro[(0)]["device_area"] 
#             self.cluster_input_area[i]        = cluster_input_ADC_area + cluster_input_Macro_area
#     
# 
#             self.weight_bit_position_area[i]  = 0
# 
#             which_filter_ADC_area = self.which_filter_num[i] * Config.ADC[ (Config.BIT_PER_CELL, 1.28)]["area"]
#             which_filter_Macro_area = self.which_filter_num[i] * 128*128 * Config.Macro[(0)]["device_area"]
#             self.which_filter_area[i]         = which_filter_ADC_area + which_filter_Macro_area



            # 改成 BIT_PER_CELL 都用 2 來降低 Macro 數
    #         self.which_OU_area[i]             = 0
    #         NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
    #         cluster_input_ADC_area = self.cluster_input_num[i] * Config.ADC[ (2, 1.28)]["area"]
    #         cluster_input_Macro_area = self.cluster_input_num[i] * 128*128 *  Config.Macro[(0)]["device_area"] 
    #         self.cluster_input_area[i]        = cluster_input_ADC_area + cluster_input_Macro_area
    # 
    #         self.weight_bit_position_area[i]  = 0
    #         
    #         which_filter_ADC_area = self.which_filter_num[i] * Config.ADC[ (2, 1.28)]["area"]
    #         which_filter_Macro_area = self.which_filter_num[i] * 128*128 * Config.Macro[(0)]["device_area"]
    #         self.which_filter_area[i]         = which_filter_ADC_area + which_filter_Macro_area

            
            # 改成等效上總共多少 cell / (128*128) 得到 #Macro, 且 BIT_PER_CELL 都用 2
            NUM_CLUSTER = math.ceil(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i] / mywork.OU)
            self.which_OU_area[i]               = 0
            self.cluster_input_area[i]          = self.cluster_input_num[i] * (Config.ADC[ (2, 1.28)]["area"] + 128**2 * Config.Macro[(0)]["device_area"])
            self.weight_bit_position_area[i]    = 0
            self.which_filter_area[i]           = self.which_filter_num[i] * (Config.ADC[ (2, 1.28)]["area"] + 128**2 * Config.Macro[(0)]["device_area"])


            ###################




            self.Macro_area[i] = self.Macro_num[i] * Config.Macro[(0)]["device_area"] * Config.XB_paramX * Config.XB_paramX
            self.SandH_area[i] = self.SandH_num[i] * Config.SandH[(0)]["area"]
            self.ADC_area[i] = self.ADC_num[i] * Config.ADC[ (mywork.ADC_PRECISION, Config.ADC_GSps)]["area"]
            
            self.Shift_and_Add_area[i] = self.Shift_and_Add_num[i] * Config.Shift_and_Add[\
                (Config.CLK_PERIOD, Config.BIT_PER_CELL, Config.BIT_DAC, mywork.OU, Config.NETWORK_DICT["BIT_W"],  Config.NETWORK_DICT["BIT_IFM"])]["area"]
            

            # ===============================================================
            # Add_and_Distributor
            # ===============================================================
            # V1 正確版
            # BIT_INPUT_SHIFT = mywork.ADC_PRECISION  + Config.NETWORK_DICT["BIT_IFM"] + Config.NETWORK_DICT["W"] -1
            # self.Decoder_area[i]    = self.Decoder_num[i] * Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["area"]
            # self.MUX_area[i]        = self.MUX_num[i] * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, BIT_INPUT_SHIFT)]["area"]
            # self.Add_area[i]        = self.Add_num[i] * Config.Add[(Config.CLK_PERIOD, \
            #                             Config.NETWORK_DICT["BIT_IFM"] + Config.NETWORK_DICT["W"] + math.ceil(math.log(Config.NETWORK_DICT["K"][i]*Config.NETWORK_DICT["IN_CH"][i], 2)))]["area"]
            

            '''
            # V2 改成固定用 MUX BIT=32, Add BIT=29, Add_Input_num 用 16 去 decompose
            BIT_INPUT_SHIFT = mywork.ADC_PRECISION  + Config.NETWORK_DICT["BIT_IFM"] + Config.NETWORK_DICT["BIT_W"] -1
            self.Decoder_area[i]    = self.Decoder_num[i] * Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["area"]
            # self.MUX_area[i]        = self.MUX_num[i] * Config.Mux_base[(Config.CLK_PERIOD, Config.Mux_base_Small_input, 32)]["area"] 
            self.MUX_area[i]        = 0
            self.Add_area[i]        = self.Add_num[i] * Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["area"]
            '''
            
            # V3  Adder_mask
            BIT_INPUT_SHIFT = mywork.ADC_PRECISION  + Config.NETWORK_DICT["BIT_IFM"] + Config.NETWORK_DICT["BIT_W"] -1
            self.Decoder_area[i]    = self.Decoder_num[i] * Config.Decoder[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i])]["area"]
            self.MUX_of_Shift_and_Add_area[i] = self.MUX_of_Shift_and_Add_num[i] * Config.Mux_base[(Config.CLK_PERIOD, Config.NETWORK_DICT["BIT_IFM"], 21)]["area"]

            # Adder 理論上是隨著 Macro數/BIT_FIM 而變得，但是我懶得改了
            self.Adder_mask_area[i] = self.Adder_mask_num[i] * Config.Adder_mask[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, Config.NETWORK_DICT["OUT_CH"][i])]["area"]

            # ex. PE0 16 Macro, PE1 32 Macro, PE2 48 Macro
            # 除以 BIT_IFM 後
            # 得到 PE0 需要 2 個 2-input Add
            # 得到 PE1 需要 4 個 2-input Add
            # 得到 PE2 需要 6 個 2-input Add
            self.Add_area[i] = sum([\
                    math.ceil(mywork.PE_num_Macro[i][j]/Config.NETWORK_DICT["BIT_IFM"]) \
                    * Interpolate(\
                        upper_level=Config.MAX_num_Macro_per_Tile, \
                        lower_level=2, \
                        your_level=math.ceil(mywork.PE_num_Macro[i][j]/Config.NETWORK_DICT["BIT_IFM"]), \
                        upper_value=Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["area"], \
                        lower_value=Config.Add_base[(Config.CLK_PERIOD, Config.MAX_num_Macro_per_Tile, 21, 28)]["area"]) for j in range(Config.NETWORK_DICT["K"][i])])\
                    * Config.NETWORK_DICT["K"][i]

            self.MUX_of_filter_area[i] = self.MUX_of_filter_num[i] * Config.Mux_base[(Config.CLK_PERIOD, Config.NETWORK_DICT["OUT_CH"][i],\
                math.ceil(math.log(Config.NETWORK_DICT["IN_CH"][i]*(Config.NETWORK_DICT["K"][i]**2),2))+Config.NETWORK_DICT["BIT_IFM"]+Config.NETWORK_DICT["BIT_W"]-1)]["area"]

            ###################
        

            self.Accumulator_area[i] = self.Accumulator_num[i] * Config.Add_base[(Config.CLK_PERIOD, Config.NETWORK_DICT["K"][i], 28, 29)]["area"]


            # total area

            self.total_area_CONV[i] \
                = self.Activation_area[i] + self.Row_Buffers_area[i] \
                + self.which_OU_area[i] + self.cluster_input_area[i] + self.weight_bit_position_area[i] + self.which_filter_area[i] \
                + self.Macro_area[i] \
                + self.SandH_area[i] + self.ADC_area[i] + self.Shift_and_Add_area[i]\
                + self.Decoder_area[i] +self.MUX_of_Shift_and_Add_area[i] + self.Adder_mask_area[i] + self.Add_area[i] + self.MUX_of_filter_area[i]\
                + self.Accumulator_area[i]

            ###################

    def get_area(self):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):

            logger.info(f" CONV{i+1}: ")
            logger.info(f"     Activation             num = {self.Activation_num[i]:>20}")
            logger.info(f"                            area = {self.Activation_area[i]:>20}um^2")

            logger.info(f"     Row_Buffers           num = {self.Row_Buffers_num[i]:>20}")
            logger.info(f"                           area = {self.Row_Buffers_area[i]:>20}um^2")


            logger.info(f"     OU_Table              ")
            logger.info(f"            which_OU            num = {self.which_OU_num[i]:>20}")
            logger.info(f"                                area = {self.which_OU_area[i]:>20}um^2")
            logger.info(f"            cluster_input       num = {self.cluster_input_num[i]:>20}")
            logger.info(f"                                area = {self.cluster_input_area[i]:>20}um^2")
            logger.info(f"            weight_bit_position num = {self.weight_bit_position_num[i]:>20}")
            logger.info(f"                                area = {self.weight_bit_position_area[i]:>20}um^2")
            logger.info(f"            which_filter        num = {self.which_filter_num[i]:>20}")
            logger.info(f"                                area = {self.which_filter_area[i]:>20}um^2")
            logger.info(f"           OU_Table area = {self.which_OU_area[i] + self.cluster_input_area[i] + self.weight_bit_position_area[i] + self.which_filter_area[i]:>20}um^2")


            logger.info(f"     Macro              num  = {self.Macro_num[i]:>20}") 
            logger.info(f"                        area = {self.Macro_area[i]:>20}um^2") 


            logger.info(f"     SandH                num = {self.SandH_num[i]:>20}") 
            logger.info(f"                          area = {self.SandH_area[i]:>20}um^2") 


            logger.info(f"     ADC                  num = {self.ADC_num[i]:>20}") 
            logger.info(f"                          area  = {self.ADC_area[i]:>20}um^2 ")
            
            logger.info(f"     Shift_and_Add         num = {self.Shift_and_Add_num[i]:>20}")
            logger.info(f"                           area = {self.Shift_and_Add_area[i]:>20}um^2")


            logger.info(f"     Add and Distributor ")
            logger.info(f"            Decoder                 num = {self.Decoder_num[i]:>20}")
            logger.info(f"                                    area = {self.Decoder_area[i]:>20}um^2")
            logger.info(f"            MUX_of_Shift_and_Add    num = {self.MUX_of_Shift_and_Add_num[i]:>20}")
            logger.info(f"                                    area = {self.MUX_of_Shift_and_Add_area[i]:>20}um^2")
            logger.info(f"            Add                     num = {self.Add_num[i]:>20}")
            logger.info(f"                                    area ={self.Add_area[i]:>20}um^2")
            logger.info(f"            Adder_mask              num = {self.Adder_mask_num[i]:>20}")
            logger.info(f"                                    area ={self.Adder_mask_area[i]:>20}um^2")
            logger.info(f"           MUX_of_filter            num = {self.MUX_of_filter_num[i]:>20}")
            logger.info(f"                                    area = {self.MUX_of_filter_area[i]:>20}")
            logger.info(f"           Add and Distributor area = {self.Decoder_area[i] +self.MUX_of_Shift_and_Add_area[i] + self.Adder_mask_area[i] + self.Add_area[i] + self.MUX_of_filter_area[i]}")


            logger.info(f"    Accumulator             num           = {self.Accumulator_num[i]:>20}")
            logger.info(f"                            area          = {self.Accumulator_area[i]:>20}")
            
            logger.info(f"     total_area = {self.total_area_CONV[i]:>20}um^2")



    
    
class Pooling_Area_Model():
    def __init__(self, mywork):

        # 初始化硬體個數的參數
        self.Pooling_num                 = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
       

        # 初始化硬體area的參數
        self.Pooling_area                = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]

     
        #
        self.calculate_num(mywork)
        self.calculate_area()
        self.get_area()

    def calculate_num(self, mywork):

        for i in range(Config.NETWORK_DICT["total_layer_num"]):

            if Config.NETWORK_DICT["POOLING_SIZE"][i]!=0 : # 代表這層 CONV 後面才有接 Pooling, 其中一般CONVi 的 output 是 CONVi+1 負責 Pool，最後CONV 的 Pooling 還是由最後 Pool

                NUM_DATA = Config.NETWORK_DICT["OUT_CH"][i] * Config.NETWORK_DICT["K"][i]
                NUM_CYCLE = math.floor(mywork.BOTTLENECK_LATENCY /  Config.NETWORK_DICT["DELTA_X"][i]) * Config.NETWORK_DICT["POOLING_STRIDE"][i] 

                if(i==Config.NETWORK_DICT["total_layer_num"]-1): # 最後一層
                    # self.Pooling_num[i] = mywork.num_Tile[i] * mywork.DUPLICATE[i] * math.ceil(NUM_DATA/NUM_CYCLE)
                    self.Pooling_num[i] = mywork.num_Tile[i] * math.ceil(NUM_DATA/NUM_CYCLE)
                else:
                    # self.Pooling_num[i] = mywork.num_Tile[i+1] * mywork.DUPLICATE[i+1] * math.ceil(NUM_DATA/NUM_CYCLE)
                    self.Pooling_num[i] = mywork.num_Tile[i+1] * math.ceil(NUM_DATA/NUM_CYCLE)
            else:
                self.Pooling_num[i] = 0
                

        
    def calculate_area(self):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            if Config.NETWORK_DICT["POOLING_SIZE"][i]!=0 : # 代表這層 CONV 後面才有接 Pooling, 其中一般CONVi 的 output 是 CONVi+1 負責 Pool，最後CONV 的 Pooling 還是由最後 Pool
                self.Pooling_area[i] = self.Pooling_num[i] * \
                    Config.Pooling[(Config.CLK_PERIOD, Config.NETWORK_DICT["POOLING_SIZE"][i], Config.NETWORK_DICT["BIT_IFM"])]["area"]
            else:
                self.Pooling_area[i] = 0
        

    def get_area(self):
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            logging.info(f" CONV{i+1}: ")
            logging.info(f"    Pooling num = {self.Pooling_num[i]:>20}um^2")
            logging.info(f"            area = {self.Pooling_area[i]:>20}um^2")


class Router_Area_Model():
    def __init__(self, mywork):
        # 初始化硬體個數的參數
        self.inter_CONV_Router_num        = 0
        self.intra_CONV_Router_num        = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
       
        # 初始化硬體area的參數
        self.inter_CONV_Router_area        = 0
        self.intra_CONV_Router_area        = [0 for i in range(Config.NETWORK_DICT["total_layer_num"])]
        self.total_area_Router             = 0

        #
        self.calculate_num(mywork)
        self.calculate_area()
        self.get_area()

    def calculate_num(self, mywork):

        # 看幾層CONV 對應 H-Tree 需要的 Router
        # ex. total_layer_num = 13
        # ceiling( log(13,2) ) = 4
        # (1 + 2 + 4 + 8 + 16) - (16-3) = 2^(4+1)-1 - (2^4 - 13)
        self.inter_CONV_Router_num = 2**(math.ceil(math.log(Config.NETWORK_DICT["total_layer_num"],2))+1)-1 \
            - (2**(math.ceil(math.log(Config.NETWORK_DICT["total_layer_num"],2))) -  Config.NETWORK_DICT["total_layer_num"])
        

        # 每層 CONV 內部的 DUPLICATE * num_Tile
        # ex. CONV1 是 4
        # log(4,2) = 2
        # (2+4) = 2^(2+1) - 2  (-2 是因為 Root 已經算在上面的CONV 對應 的 H-Tree 那邊了)
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            # self.intra_CONV_Router_num[i] = (2**(math.ceil(math.log(mywork.DUPLICATE[i],2)) +1) - 2) \
            #     - (2**(math.ceil(math.log(mywork.DUPLICATE[i],2))) - mywork.DUPLICATE[i])


            self.intra_CONV_Router_num[i] = (2**(math.ceil(math.log(mywork.num_Tile[i],2)) +1) - 2) \
                - (2**(math.ceil(math.log(mywork.num_Tile[i],2))) - mywork.num_Tile[i])


    def calculate_area(self):
        self.inter_CONV_Router_area = self.inter_CONV_Router_num * Config.Router[(Config.CLK_PERIOD, 29)]["area"]

        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            self.intra_CONV_Router_area[i] = self.intra_CONV_Router_num[i] * Config.Router[(Config.CLK_PERIOD, 29)]["area"]
        
        self.total_area_Router = self.inter_CONV_Router_area + sum(self.intra_CONV_Router_area)


    def get_area(self):
        logging.info(f" inter_CONV_Router_area= {self.inter_CONV_Router_area:>20}um^2")
        for i in range(Config.NETWORK_DICT["total_layer_num"]):
            logging.info(f" CONV{i+1}: ")
            logging.info(f"    intra CONV  num =  {self.intra_CONV_Router_num[i]:>20}um^2")
            logging.info(f"                area = {self.intra_CONV_Router_area[i]:>20}um^2")
        logging.info(f" total Router area = {self.total_area_Router:>20}um^2")


