


CLK_PERIOD =  0.78125 # 0.78125(因為1/1.28), 3.125, 4.6875, 7.8125, 14.0625, 26.5625 # 10 # 單位 : ns
ADC_GSps = 1.28
LATENCY = (5.882*1e+06 / CLK_PERIOD )# 單位 : 幾個 cycle, ex. 10ms / 3.125 = 10*1e+06 ns / 3.125 ns = 3200000 cycles 
# OU = [2, 4, 8, 16, 32]
OU = 2
Mux_base_Small_input = 32
# Add_and_Distributor_Small_input = 16
XB_paramX = 128
BIT_DAC = 1
BIT_PER_CELL = 2
MAX_num_Macro_per_Tile = 16 # 這邊是因為 Add and Distributor 那邊的 Add 最多能吃 16 input 訂的



# block_middle_time 指的是 最後一層 convolution 在算的時候 Tile 得出一組 OFM rows 的次數
# ex. CONV11 算 OFM row : [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14]] 總共 5 個 block
# 然後因為 padding 關係，所以相當於 CONV12 IFM row 的 [[2,3,4], [4,5,6].... ，另外最後一個 row 也有 padding
# 在這段時間 CONV10  在算 OFM row  [[6,7,8], [9,10,11], ......
# POOLING_STRIDE 跟 SIZE 是 0 的代表那層 convolution 後面沒有接 Pooling
NETWORK_DICT = { \
    "total_layer_num"   : 13,\
    "OFM_row"           :[224,224,112,112,56,56,56,28,28,28,14,14,14],\
    # "OFM_row"           :[32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2],\
    "IFM_row"           :[226,226,114,114,58,58,58,30,30,30,16,16,16],\
    
    
    "BLOCK_MIDDLE_TIME" :[ 41,41,  22,22,  12,12,12, 7,8,8,  5,5,5],\
    # "BLOCK_MIDDLE_TIME" :[ 4,4,2,2,1,1,1,  1,1,1,1,1,1],\


    "DELTA_X"           :[16,16,8,8,4,4,4,2,2,2,1,1,1],\
    
    "IN_CH"             :[3,64,64,128,128,256,256,256,512,512,512,512,512],\
    "OUT_CH"            :[64,64,128,128,256,256,256,512,512,512,512,512,512],\
    
    
    "K"                 :[ 3, 3,3,3,3,3,3,3,3,3,3,3,3], \
    "CONV_STRIDE"       :[ 1, 1,1,1,1,1,1,1,1,1,1,1,1],\
    "POOLING_SIZE"      :[ 0, 2,0,2,0,0,2,0,0,2,0,0,2], \
    "POOLING_STRIDE"    :[ 0, 2,0,2,0,0,2,0,0,2,0,0,2], \
    "BIT_IFM"           :8, \
    "BIT_W"             :8, \
    
    "SRAM_NUM_WORD" : [448, 2400, 1792, 2400, 1792, 2400, 2400, 1792, 2400, 2400, 1792, 1792, 1792], \
    "SRAM_WIDTH"    : [12, 48, 32, 48, 32, 48, 48, 32, 48, 48, 32, 32, 32] \
    # "SRAM_NUM_WORD" : [256, 1024, 512, 1024, 512, 1024, 1024, 512, 1024, 1024, 512, 512, 512], \
    # "SRAM_WIDTH"    : [8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16] \
}



# area 單位是 um^2
# power 單位是 1W

# 先用 MNSIM 數據, 單位 1W, um^2, ns
Macro = {
    
    # (0) : {"device_power": 2.2020833333333332e-07, "device_area": 1.44, "crossbar_read_latency" : 3.16} # MNSIM(45nm)
    # (0) : {"device_power": 2.2020833333333332e-07, "device_area": 0.00152, "crossbar_read_latency" : 3.16} # MNSIM power, ISAAC area, MNSIM latency
    # (0) : {"device_power": 1.6997e-08, "device_area": 0.0183, "crossbar_read_latency": 3.16} # Optimizing Weight Mapping(32nm), 用 0.88pJ/op 除以 read latency 回推 power, read latency 不太重要

    (0) : {"device_power": 1.458860759493671e-6, "device_area": 0.411, "crossbar_read_latency" : 3.16} # 3.16 隨便的， device energy 是 461fJ # A 40nm 64kb 26.56TOPS/W 2.37Mb/mm2 RRAM Binary/Compute-in-Memory Macro with 4.23× Improvement in Density and >75% Use of Sensing Dynamic Range 


}

# CLK_PERIOD, BIT_IFM, BIT_OFM
Activation = {
    (0.78125, 8, 29) : {"area": 99.338396, "power": 9.9684e-05},
    (3.125, 8, 29) : {"area": 99.338396 , "power": 2.769e-05},
    (4.6875, 8, 29) : {"area":99.338396 , "power": 1.867e-05},
    (7.8125, 8, 29) : {"area":99.338396 , "power": 1.147e-05},
    (14.0625, 8, 29) : {"area":99.338396 , "power": 6.664e-06},
    (26.5625,8, 29) : {"area":99.338396 , "power": 3.839e-06}
}

# CLK_PERIOD, SRAM_NUM_WORD, SRAM_WIDTH
Row_Buffers = {
    (0.78125, 448, 12) : {"area": 49764.980762, "power": 25.7832e-03},
    (3.125, 448, 12) : {"area":49764.980762 , "power": 7.162e-03 },
    (4.6875, 448, 12) : {"area": 49764.527162 , "power": 4.891e-03},
    (7.8125, 448, 12) : {"area":49766.795162 , "power": 3.074e-03},
    (14.0625, 448, 12) : {"area": 49766.795162 , "power": 1.863e-03},
    (26.5625, 448, 12) : {"area":49766.795162 , "power": 1.150e-03},

    (0.78125, 1792, 32) : {"area":157141.425963 , "power":  0.05652},
    (3.125, 1792, 32) : {"area": 157141.425963 , "power":  0.0157},
    (4.6875, 1792, 32) : {"area":157142.333163 , "power":  0.0109},
    (7.8125, 1792, 32) : {"area":157142.333163 , "power": 7.022e-03},
    (14.0625, 1792, 32) : {"area":157152.539164 , "power": 4.448e-03},
    (26.5625, 1792, 32) : {"area":157153.219564 , "power": 2.935e-03 },
    
    (0.78125, 2400, 48) : {"area": 281223.757494, "power": 0.08316},
    (3.125, 2400, 48) : {"area": 281223.757494, "power":  0.0231},
    (4.6875, 2400, 48) : {"area": 281225.798694 , "power": 0.0162},
    (7.8125, 2400, 48) : {"area": 281234.870695, "power": 0.0106},
    (14.0625, 2400, 48) : {"area": 281229.881095, "power": 6.829e-03},
    (26.5625, 2400, 48) : {"area": 281229.200695, "power": 4.637e-03},


    (3.125,   256, 8) : {"area":  43214.799449, "power": 4.270e-04}, 
    (4.6875,  256, 8) : {"area":  43215.253049, "power": 2.872e-04}, 
    (7.8125,  256, 8) : {"area": 43215.253049, "power": 2.0e-04},
    (14.0625, 256, 8) : {"area": 43215.253049, "power": 1.3e-04},
    (26.5625, 256, 8) : {"area": 43215.253049, "power": 6.2e-05},

    (3.125,   1024, 16) : {"area": 68946.728960, "power": 8.402e-03},
    (4.6875,  1024, 16) : {"area": 68946.728960, "power": 6.0e-03},
    (7.8125,  1024, 16) : {"area": 68946.728960, "power": 4.0e-03},
    (14.0625, 1024, 16) : {"area": 68946.728960, "power": 2.0e-03},
    (26.5625, 1024, 16) : {"area": 68947.636160, "power": 1.473e-03},
    
    (3.125,   512, 16) : {"area": 52258.862093, "power": 7.610e-03},
    (4.6875,  512, 16) : {"area": 52258.862093, "power": 5.0e-03},
    (7.8125,  512, 16) : {"area": 52258.862093, "power": 2.5e-03},
    (14.0625, 512, 16) : {"area": 52258.862093, "power": 1.6e-03},
    (26.5625, 512, 16) : {"area": 52258.862093, "power": 1.242e-03}

    
}


# CLK_PERIOD, MUX_NUM_INPUT, BIT
# which_OU={
#     (10, ) : {"area": , "power": }
# }
# # CLK_PERIOD, MUX_NUM_INPUT, BIT
# cluster_input={
#     () : {"area": , "power": }
# }
# # CLK_PERIOD, MUX_NUM_INPUT, BIT
# weight_bit_position={
#     () : {"area": , "power": }
# }
# # CLK_PERIOD, MUX_NUM_INPUT, BIT
# which_filter={
#     () : {"area": , "power": }
# }


# only one data
SandH = {
    (0) : {"area": 0.04, "power": 9.765625e-9}
}


# ADC_PRECISION, GSps(giga samples per second)
ADC = {
    # 照著 Visio/OU_Table.vsdx 公式
    # (1, 1.28): {"area": 79.6875, "power": 7.031250000000001e-05},
    # (2, 1.28): {"area": 159.375, "power": 9.375e-05},
    # (3, 1.28): {"area": 243.75,  "power": 0.00014062500000000002},
    # (4, 1.28): {"area": 337.5,   "power": 0.00022500000000000002},
    # (5, 1.28): {"area": 450.0,   "power": 0.000375},
    # (6, 1.28): {"area": 600.0,   "power": 0.0006428571428571429},
    # (7, 1.28): {"area": 825.0,   "power": 0.0011250000000000001},
    # (8, 1.28): {"area": 1200.0,   "power": 0.002}

    # 照著 Visio/OU_Table.vsdx 公式
    (1, 1.28): {"area": 79.6875, "power": 1.5625e-05},
    (2, 1.28): {"area": 159.375, "power": 3.125e-05},
    (3, 1.28): {"area": 243.75,  "power": 6.25e-05},
    (4, 1.28): {"area": 337.5,   "power": 0.000125},
    (5, 1.28): {"area": 450.0,   "power": 0.00025},
    (6, 1.28): {"area": 600.0,   "power": 0.0005},
    (7, 1.28): {"area": 825.0,   "power": 0.001},
    (8, 1.28): {"area": 1200.0,   "power": 0.002}
}


# CLK_PERIOD, BIT_PER_CELL, BIT_DAC, OU, BIT_W,  BIT_IFM
Shift_and_Add = {

    # 正確版
#     (3.125, 1, 1, 2, 8, 8): {"area": 402.569984 , "power": 8.700e-05},
#     (4.6875, 1, 1, 2, 8, 8): {"area": 402.569984 , "power": 5.890e-05 },
#     (7.8125, 1, 1, 2, 8, 8): {"area":  402.569984 , "power": 3.642e-05 },
#     (14.0625, 1, 1, 2, 8, 8): {"area": 402.569984, "power": 2.144e-05},
#     (26.5625, 1, 1, 2, 8, 8): {"area": 402.569984, "power": 1.263e-05},    
# 
#     (3.125,   2, 1, 2, 8, 8): {"area": 474.238783 , "power": 9.721e-05},
#     (4.6875,  2, 1, 2, 8, 8): {"area": 474.238783 , "power": 6.593e-05 },
#     (7.8125,  2, 1, 2, 8, 8): {"area": 474.238783 , "power": 4.091e-05 },
#     (14.0625, 2, 1, 2, 8, 8): {"area": 474.238783,  "power": 2.423e-05},
#     (26.5625, 2, 1, 2, 8, 8): {"area": 474.238783,  "power": 1.442e-05},
# 
#     
#     
#     (3.125,   1, 1, 4, 8, 8): {"area": 474.238783, "power": 9.721e-05 },
#     (4.6875,  1, 1, 4, 8, 8): {"area": 474.238783, "power": 6.593e-05},
#     (7.8125,  1, 1, 4, 8, 8): {"area": 474.238783, "power": 4.091e-05},
#     (14.0625, 1, 1, 4, 8, 8): {"area": 474.238783, "power": 2.423e-05},
#     (26.5625, 1, 1, 4, 8, 8): {"area": 474.238783, "power": 1.442e-05},
# 
#     (3.125,   2, 1, 4, 8, 8): {"area": 547.268380, "power": 1.067e-04 },
#     (4.6875,  2, 1, 4, 8, 8): {"area": 547.268380, "power": 7.248e-05},
#     (7.8125,  2, 1, 4, 8, 8): {"area": 547.268380, "power": 4.510e-05},
#     (14.0625, 2, 1, 4, 8, 8): {"area": 547.268380, "power": 2.684e-05},
#     (26.5625, 2, 1, 4, 8, 8): {"area": 547.268380, "power": 1.610e-05},
# 
# 
#     
#     (3.125, 1, 1, 8, 8, 8): {"area":   547.268380,   "power": 1.067e-04 },
#     (4.6875, 1, 1, 8, 8, 8): {"area":  547.268380,  "power": 7.248e-05},
#     (7.8125, 1, 1, 8, 8, 8): {"area":  547.268380,  "power": 4.510e-05},
#     (14.0625, 1, 1, 8, 8, 8): {"area": 547.268380, "power": 2.684e-05},
#     (26.5625, 1, 1, 8, 8, 8): {"area": 547.268380, "power": 1.610e-05},
# 
#     (3.125,   2, 1, 8, 8, 8): {"area": 624.153578, "power": 1.171e-04 },
#     (4.6875,  2, 1, 8, 8, 8): {"area": 624.153578, "power": 9.23e-05},
#     (7.8125,  2, 1, 8, 8, 8): {"area": 624.153578, "power": 6.75e-05},
#     (14.0625, 2, 1, 8, 8, 8): {"area": 624.153578, "power": 4.27e-05},
#     (26.5625, 2, 1, 8, 8, 8): {"area": 624.153578, "power": 1.790e-05},
# 
# 
#     (3.125, 1, 1, 16, 8, 8): {"area":   624.153578,   "power": 1.171e-04},
#     (4.6875, 1, 1, 16, 8, 8): {"area":  624.153578,  "power": 9.23e-05},
#     (7.8125, 1, 1, 16, 8, 8): {"area":  624.153578,  "power": 6.75e-05},
#     (14.0625, 1, 1, 16, 8, 8): {"area": 624.153578, "power": 4.27e-05},
#     (26.5625, 1, 1, 16, 8, 8): {"area": 624.153578,"power": 1.790e-05},
# 
#     (3.125,   2, 1, 16, 8, 8): {"area": 692.420375,  "power": 1.278e-04},
#     (4.6875,  2, 1, 16, 8, 8): {"area": 692.42037,  "power": 0.00010078999999999999},
#     (7.8125,  2, 1, 16, 8, 8): {"area": 692.42037,  "power": 7.377999999999999e-05},
#     (14.0625, 2, 1, 16, 8, 8): {"area": 692.42037,  "power": 4.677e-05},
#     (26.5625, 2, 1, 16, 8, 8): {"area": 692.420375, "power": 1.976e-05},
#     
# 
# 
#     (3.125, 1, 1, 32, 8, 8): {"area":   692.420375,  "power": 1.278e-04},
#     (4.6875, 1, 1, 32, 8, 8): {"area":  692.42037,  "power": 0.00010078999999999999},
#     (7.8125, 1, 1, 32, 8, 8): {"area":  692.42037,  "power": 7.377999999999999e-05},
#     (14.0625, 1, 1, 32, 8, 8): {"area": 692.42037, "power": 4.677e-05},
#     (26.5625, 1, 1, 32, 8, 8): {"area": 692.420375,"power": 1.976e-05},
# 
#     (3.125,   2, 1, 32, 8, 8): {"area": 700.0836,  "power": 1.32e-04},
#     (4.6875,  2, 1, 32, 8, 8): {"area": 700.0836,  "power": 0.0001040325},
#     (7.8125,  2, 1, 32, 8, 8): {"area": 700.0836,  "power": 7.6065e-05},
#     (14.0625, 2, 1, 32, 8, 8): {"area": 700.0836,  "power": 4.8097500000000005e-05},
#     (26.5625, 2, 1, 32, 8, 8): {"area": 700.0836,  "power": 2.013e-05}


    # paper 估計版, *0.2
    (0.78125, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 6.264e-05},
    (3.125, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 1.74e-05}, 
    (4.6875, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 1.178e-05}, 
    (7.8125, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 7.284e-06}, 
    (14.0625, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 4.288e-06}, 
    (26.5625, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 2.526e-06}, 
    
    (0.78125, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 6.99912e-05},
    (3.125, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 1.9442e-05}, 
    (4.6875, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 1.3186e-05}, 
    (7.8125, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 8.182e-06}, 
    (14.0625, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 4.846000000000001e-06}, 
    (26.5625, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 2.884e-06}, 
    
    (0.78125, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 6.99912e-05},
    (3.125, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 1.9442e-05}, 
    (4.6875, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 1.3186e-05}, 
    (7.8125, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 8.182e-06}, 
    (14.0625, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 4.846000000000001e-06}, 
    (26.5625, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 2.884e-06}, 
    
    (0.78125, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 7.6824e-05},
    (3.125, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 2.1340000000000002e-05}, 
    (4.6875, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 1.4496e-05}, 
    (7.8125, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 9.02e-06}, 
    (14.0625, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 5.368e-06}, 
    (26.5625, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 3.2199999999999997e-06}, 
    
    (0.78125, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 7.6824e-05},
    (3.125, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 2.1340000000000002e-05}, 
    (4.6875, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 1.4496e-05}, 
    (7.8125, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 9.02e-06}, 
    (14.0625, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 5.368e-06}, 
    (26.5625, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 3.2199999999999997e-06}, 
    
    (0.78125, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 8.4312e-05},
    (3.125, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 2.3420000000000003e-05}, 
    (4.6875, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 1.846e-05}, 
    (7.8125, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 1.3500000000000001e-05}, 
    (14.0625, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 8.540000000000001e-06}, 
    (26.5625, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 3.5800000000000005e-06}, 
    
    (0.78125, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 8.4312e-05},
    (3.125, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 2.3420000000000003e-05}, 
    (4.6875, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 1.846e-05}, 
    (7.8125, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 1.3500000000000001e-05}, 
    (14.0625, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 8.540000000000001e-06}, 
    (26.5625, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 3.5800000000000005e-06}, 
    
    (0.78125, 2, 1, 16, 8, 8): {'area': 692.420375, 'power': 9.2016e-05},
    (3.125, 2, 1, 16, 8, 8): {'area': 692.420375, 'power': 2.556e-05}, 
    (4.6875, 2, 1, 16, 8, 8): {'area': 692.42037, 'power': 2.0158e-05},
    (7.8125, 2, 1, 16, 8, 8): {'area': 692.42037, 'power': 1.4755999999999999e-05}, 
    (14.0625, 2, 1, 16, 8, 8): {'area': 692.42037, 'power': 9.354e-06}, 
    (26.5625, 2, 1, 16, 8, 8): {'area': 692.420375, 'power': 3.9520000000000004e-06}, 
    
    (0.78125, 1, 1, 32, 8, 8): {'area': 692.420375, 'power': 9.2016e-05},
    (3.125, 1, 1, 32, 8, 8): {'area': 692.420375, 'power': 2.556e-05}, 
    (4.6875, 1, 1, 32, 8, 8): {'area': 692.42037, 'power': 2.0158e-05}, 
    (7.8125, 1, 1, 32, 8, 8): {'area': 692.42037, 'power': 1.4755999999999999e-05}, 
    (14.0625, 1, 1, 32, 8, 8): {'area': 692.42037, 'power': 9.354e-06}, 
    (26.5625, 1, 1, 32, 8, 8): {'area': 692.420375, 'power': 3.9520000000000004e-06}, 
    
    (0.78125, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 9.504e-05},
    (3.125, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 2.6400000000000005e-05}, 
    (4.6875, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 2.08065e-05}, 
    (7.8125, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 1.5213e-05}, 
    (14.0625, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 9.619500000000002e-06}, 
    (26.5625, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 4.026e-06}

}


# CLK_PERIOD, NUM_OUTPUT
Decoder ={
    # 正確版
#     (3.125, 64): {"area": 405.064785, "power": 9.046e-05},
#     (4.6875, 64): {"area": 405.064785, "power": 7.108749999999999e-05},
#     (7.8125, 64): {"area": 405.064785, "power": 5.1714999999999995e-05},
#     (14.0625, 64): {"area": 405.064785,"power": 3.23425e-05},
#     (26.5625, 64): {"area": 405.064785,"power": 1.297e-05},
# 
#     (3.125, 128): {"area":  752.068771,"power": 1.671e-04},
#     (4.6875, 128): {"area": 752.06877,"power": 1.313325e-04},
#     (7.8125, 128): {"area": 752.06877,"power": 9.5565e-05},
#     (14.0625, 128): {"area":752.06877 ,"power": 5.97975e-05},
#     (26.5625, 128): {"area": 752.068771,"power": 2.403e-05},
# 
#     (3.125, 256): {"area": 1437.685144,"power": 3.185e-04},
#     (4.6875, 256): {"area": 1436.551144,"power": 2.503325e-04},
#     (7.8125, 256): {"area": 1436.551144,"power":  1.821649999e-04},
#     (14.0625, 256): {"area":1436.551144 ,"power":  1.13997e-04},
#     (26.5625, 256): {"area": 1436.551144,"power": 4.583e-05},
# 
# 
#     (3.125, 512): {"area":  2809.598290,"power": 6.201e-04 },
#     (4.6875, 512): {"area": 2809.144690,"power": 0.00048744},
#     (7.8125, 512): {"area": 2809.144690,"power": 0.00035477999},
#     (14.0625, 512): {"area":2809.144690 ,"power":  0.00022212},
#     (26.5625, 512): {"area":  2809.144690,"power": 8.946e-05}


    # *0.3 版
    (0.78125, 64) : {'area': 405.064785, 'power': 10.647e-05},
    (3.125, 64) : {'area': 405.064785, 'power': 2.7138000000000003e-05},
    (4.6875, 64) : {'area': 405.064785, 'power': 2.1326249999999996e-05},
    (7.8125, 64) : {'area': 405.064785, 'power': 1.55145e-05},
    (14.0625, 64) : {'area': 405.064785, 'power': 9.702749999999998e-06},
    (26.5625, 64) : {'area': 405.064785, 'power': 3.891e-06},
    
    (0.78125, 128) : {'area': 752.068771, 'power': 18.046799964e-05},
    (3.125, 128) : {'area': 752.068771, 'power': 5.0129999999999996e-05},
    (4.6875, 128) : {'area': 752.06877, 'power': 3.939975e-05},
    (7.8125, 128) : {'area': 752.06877, 'power': 2.86695e-05},
    (14.0625, 128) : {'area': 752.06877, 'power': 1.793925e-05},
    (26.5625, 128) : {'area': 752.068771, 'power': 7.208999999999998e-06},
    
    (0.78125, 256) : {'area': 1437.685144, 'power': 34.3979964e-05},
    (3.125, 256) : {'area': 1437.685144, 'power': 9.554999999999999e-05},
    (4.6875, 256) : {'area': 1436.551144, 'power': 7.509974999999999e-05},
    (7.8125, 256) : {'area': 1436.551144, 'power': 5.4649499969999996e-05},
    (14.0625, 256) : {'area': 1436.551144, 'power': 3.4199099999999996e-05},
    (26.5625, 256) : {'area': 1436.551144, 'power': 1.3749000000000002e-05},
    

    (0.78125, 512) : {'area': 2809.59829, 'power': 0.0006697079964},
    (3.125, 512) : {'area': 2809.59829, 'power': 0.00018602999999999998},
    (4.6875, 512) : {'area': 2809.14469, 'power': 0.000146232},
    (7.8125, 512) : {'area': 2809.14469, 'power': 0.00010643399700000002},
    (14.0625, 512) : {'area': 2809.14469, 'power': 6.663599999999999e-05},
    (26.5625, 512) : {'area': 2809.14469, 'power': 2.6838e-05}
}


# CLK_PERIOD, NUM_INPUT, BIT
# 有 which_OU, cluster_input, weight_bit_position, which_filter, Shift and Add 會需要
Mux_base = {

    # 正確版
#     (3.125, 32, 32) : {"area": 1006.538354, "power":  4.290e-04},
#     (4.6875, 32, 32) : {"area": 1005.63115, "power": 0.00033590},
#     (7.8125, 32, 32) : {"area": 1005.63115, "power": 0.0002428},
#     (14.0625, 32, 32) : {"area":1005.63115 , "power": 0.0001497},
#     (26.5625, 32, 32) : {"area": 1005.631154, "power": 5.660e-05},
# 
#     (3.125,   3, 20) : {"area": 130.863595, "power": 1.079e-04},
#     (4.6875,  3, 20) : {"area": 130.863595, "power": 8.429749999999999e-05},
#     (7.8125,  3, 20) : {"area": 130.863595, "power": 6.0695e-05},
#     (14.0625, 3, 20) : {"area": 130.863595, "power": 3.70925e-05},
#     (26.5625, 3, 20) : {"area": 130.863595, "power": 1.349e-05},
#     
#     (3.125,   64, 25) : {"area": 1449.705543, "power": 2.342e-04},
#     (4.6875,  64, 25) : {"area": 1449.705543, "power": 0.00018491999999999998},
#     (7.8125,  64, 25) : {"area": 1449.705543, "power": 0.00013564},
#     (14.0625, 64, 25) : {"area": 1449.705543, "power": 8.636e-05},
#     (26.5625, 64, 25) : {"area": 1449.705543, "power": 3.708e-05},
# 
#     (3.125,   128, 26) : {"area": 2844.752286, "power":  3.549e-04},
#     (4.6875,  128, 26) : {"area": 2844.752286, "power": 0.0002812225},
#     (7.8125,  128, 26) : {"area": 2844.752286, "power": 0.000207545},
#     (14.0625, 128, 26) : {"area": 2844.752286, "power": 0.0001338675},
#     (26.5625, 128, 26) : {"area": 2844.752286, "power": 6.019e-05},
# 
#     (3.125,   256, 27) : {"area": 5693.813754, "power": 4.049e-04 },
#     (4.6875,  256, 27) : {"area": 5695.628154, "power": 0.0003247775},
#     (7.8125,  256, 27) : {"area": 5695.628154, "power": 0.000244655},
#     (14.0625, 256, 27) : {"area": 5695.628154, "power": 0.0001645325},
#     (26.5625, 256, 27) : {"area": 5695.628154, "power": 8.441e-05},
# 
#     (3.125,   512, 28) : {"area": 11586.757889, "power": 6.775e-04},
#     (4.6875,  512, 28) : {"area": 11589.933090, "power": 0.000546825},
#     (7.8125,  512, 28) : {"area": 11589.933090, "power": 0.00041615},
#     (14.0625, 512, 28) : {"area": 11589.933090, "power": 0.000285475},
#     (26.5625, 512, 28) : {"area": 11589.933090, "power": 1.548e-04 }




    # *0.3 版
    (0.78125, 8, 21)   : {'area': 229.294789, 'power': 0.00015012},
    (3.125, 8, 21)   : {'area': 229.294789, 'power': 0.0000417},
    (4.6875, 8, 21)  : {'area': 229.294789, 'power': 0.000027969},
    (7.8125, 8, 21)  : {'area': 229.294789, 'power': 0.000016977},
    (14.0625, 8, 21) : {'area': 229.294789, 'power': 0.000009651},
    (26.5625, 8, 21) : {'area': 229.294789, 'power': 0.00000534}, 

    (0.78125, 32, 32)    : {'area': 1006.538354, 'power': 0.00046332},
    (3.125, 32, 32)    : {'area': 1006.538354, 'power': 0.0001287},
    (4.6875, 32, 32)   : {'area': 1005.63115, 'power': 0.00010076999999999999},
    (7.8125, 32, 32)   : {'area': 1005.63115, 'power': 7.284e-05},
    (14.0625, 32, 32)  : {'area': 1005.63115, 'power': 4.491e-05},
    (26.5625, 32, 32)  : {'area': 1005.631154, 'power': 1.698e-05},

    (0.78125, 3, 20)     : {'area': 130.863595, 'power': 11.6532e-05},
    (3.125, 3, 20)     : {'area': 130.863595, 'power': 3.237e-05},
    (4.6875, 3, 20)    : {'area': 130.863595, 'power': 2.5289249999999995e-05},
    (7.8125, 3, 20)    : {'area': 130.863595, 'power': 1.82085e-05},
    (14.0625, 3, 20)   : {'area': 130.863595, 'power': 1.1127749999999999e-05},
    (26.5625, 3, 20)   : {'area': 130.863595, 'power': 4.0469999999999995e-06},
    
    (0.78125, 64, 20) : {'area': 1144.659548, 'power': 24.7212e-05},
    (3.125, 64, 20) : {'area': 1144.659548, 'power': 6.867e-05},
    (4.6875, 64, 20) : {'area': 1144.659548, 'power': 5.4071999999999986e-05},
    (7.8125, 64, 20) : {'area': 1144.659548, 'power': 3.9474e-05},
    (14.0625, 64, 20) : {'area': 1144.659548, 'power': 2.4876e-05},
    (26.5625, 64, 20) : {'area': 1144.886348, 'power': 1.0278e-05},


    (0.78125, 64, 25)    : {'area': 1449.705543, 'power': 25.2936e-05},
    (3.125, 64, 25)    : {'area': 1449.705543, 'power': 7.026e-05},
    (4.6875, 64, 25)   : {'area': 1449.705543, 'power': 5.5475999999999995e-05},
    (7.8125, 64, 25)   : {'area': 1449.705543, 'power': 4.0691999999999996e-05},
    (14.0625, 64, 25)  : {'area': 1449.705543, 'power': 2.5908e-05},
    (26.5625, 64, 25)  : {'area': 1449.705543, 'power': 1.1123999999999999e-05},
    

    (0.78125, 128, 25) : {'area': 2737.47589, 'power': 0.000360612},
    (3.125, 128, 25) : {'area': 2737.47589, 'power': 0.00010017},
    (4.6875, 128, 25) : {'area': 2737.47589, 'power': 7.940174999999999e-05},
    (7.8125, 128, 25) : {'area': 2737.47589, 'power': 5.86335e-05},
    (14.0625, 128, 25) : {'area': 2737.47589, 'power': 3.78651e-05},
    (26.5625, 128, 25) : {'area': 2737.47589, 'power': 1.7097e-05},
    
    (0.78125, 128, 26)   : {'area': 2844.752286, 'power': 0.000383292},
    (3.125, 128, 26)   : {'area': 2844.752286, 'power': 0.00010647},
    (4.6875, 128, 26)  : {'area': 2844.752286, 'power': 8.436675e-05},
    (7.8125, 128, 26)  : {'area': 2844.752286, 'power': 6.22635e-05},
    (14.0625, 128, 26) : {'area': 2844.752286, 'power': 4.016025e-05},
    (26.5625, 128, 26) : {'area': 2844.752286, 'power': 1.8057e-05},
    

    (0.78125, 256, 26) : {'area': 5476.312561, 'power': 23.92164e-05},
    (3.125, 256, 26) : {'area': 5476.312561, 'power': 6.644999999999999e-05},
    (4.6875, 256, 26) : {'area': 5475.858961, 'power': 5.591699999999999e-05},
    (7.8125, 256, 26) : {'area': 5475.858961, 'power': 4.5384e-05},
    (14.0625, 256, 26) : {'area': 5475.858961, 'power': 3.4850999999999995e-05},
    (26.5625, 256, 26) : {'area': 5475.858961, 'power': 2.4318e-05},
    

    (0.78125, 256, 27)   : {'area': 5693.813754, 'power': 0.00043729199964},
    (3.125, 256, 27)   : {'area': 5693.813754, 'power': 0.00012146999999999999},
    (4.6875, 256, 27)  : {'area': 5695.628154, 'power': 9.743325e-05},
    (7.8125, 256, 27)  : {'area': 5695.628154, 'power': 7.33965e-05},
    (14.0625, 256, 27) : {'area': 5695.628154, 'power': 4.9359750000000005e-05},
    (26.5625, 256, 27) : {'area': 5695.628154, 'power': 2.5322999999999998e-05},
    

    (0.78125, 512, 27) : {'area': 11186.455908, 'power': 0.000699624},
    (3.125, 512, 27) : {'area': 11186.455908, 'power': 0.00019434},
    (4.6875, 512, 27) : {'area': 11185.548708, 'power': 0.000156915},
    (7.8125, 512, 27) : {'area': 11185.548708, 'power': 0.00011948999999999998},
    (14.0625, 512, 27) : {'area': 11185.548708, 'power': 8.2065e-05},
    (26.5625, 512, 27) : {'area': 11185.548708, 'power': 4.464e-05},
    
    (0.78125, 512, 28)   : {'area': 11586.757889, 'power': 0.0007317},
    (3.125, 512, 28)   : {'area': 11586.757889, 'power': 0.00020325},
    (4.6875, 512, 28)  : {'area': 11589.93309, 'power': 0.0001640475},
    (7.8125, 512, 28)  : {'area': 11589.93309, 'power': 0.000124845},
    (14.0625, 512, 28) : {'area': 11589.93309, 'power': 8.56425e-05},
    (26.5625, 512, 28) : {'area': 11589.93309, 'power': 4.6439999999999996e-05}

    
}

# CLK_PERIOD,NUM_MACRO, OUT_CH
Adder_mask = {

    # 正確版
#     (3.125,   16, 64) : {"area": 1999.015, "power": 2.845e-04},
#     (4.6875,  16, 64) : {"area": 1999.015, "power": 0.00022448249999999998},
#     (7.8125,  16, 64) : {"area": 1999.015, "power": 0.00016446499999999998},
#     (14.0625, 16, 64) : {"area": 1999.015, "power": 0.00010444749999999999},
#     (26.5625, 16, 64) : {"area": 1999.015, "power": 4.443e-05},
# 
#     (3.125,   16, 128) : {"area": 2196.784731, "power": 3.010e-04},
#     (4.6875,  16, 128) : {"area": 2196.784731, "power": 2.052e-04},
#     (7.8125,  16, 128) : {"area": 2196.784731, "power": 1.285e-04},
#     (14.0625, 16, 128) : {"area": 2196.784731, "power": 7.743e-05},
#     (26.5625, 16, 128) : {"area": 2196.784731, "power": 4.737e-05},
# 
#     (3.125,   16, 256) : {"area": 2395.007925, "power": 3.190e-04},
#     (4.6875,  16, 256) : {"area": 2395.007925, "power": 0.000251915},
#     (7.8125,  16, 256) : {"area": 2395.007925, "power": 0.00018483},
#     (14.0625, 16, 256) : {"area": 2395.007925, "power": 0.00011774499999999999},
#     (26.5625, 16, 256) : {"area": 2395.007925, "power": 5.066e-05},
# 
#     (3.125,   16, 512) : {"area": 2571.685096, "power": 3.387e-04},
#     (4.6875,  16, 512) : {"area": 2571.685096, "power": 0.00026748},
#     (7.8125,  16, 512) : {"area": 2571.685096, "power": 0.00019627000000000002},
#     (14.0625, 16, 512) : {"area": 2571.685096, "power": 0.000125055},
#     (26.5625, 16, 512) : {"area": 2571.685096, "power": 5.384e-05}


    # power*0.3 / 8  版,  area / 8 版
    (0.78125, 16, 64) : {'area': 249.876875, 'power': 3.840749964e-05},
    (3.125, 16, 64) : {'area': 249.876875, 'power': 1.0668749999999998e-05},
    (4.6875, 16, 64) : {'area': 249.876875, 'power': 8.41809375e-06},
    (7.8125, 16, 64) : {'area': 249.876875, 'power': 6.167437499999999e-06},
    (14.0625, 16, 64) : {'area': 249.876875, 'power': 3.91678125e-06},
    (26.5625, 16, 64) : {'area': 249.876875, 'power': 1.6661249999999999e-06},
    

    (0.78125, 16, 128) : {'area': 274.598091375, 'power': 4.0635e-05},
    (3.125, 16, 128) : {'area': 274.598091375, 'power': 1.12875e-05},
    (4.6875, 16, 128) : {'area': 274.598091375, 'power': 7.695e-06},
    (7.8125, 16, 128) : {'area': 274.598091375, 'power': 4.81875e-06},
    (14.0625, 16, 128) : {'area': 274.598091375, 'power': 2.9036249999999997e-06},
    (26.5625, 16, 128) : {'area': 274.598091375, 'power': 1.776375e-06},
    

    (0.78125, 16, 256) : {'area': 299.375990625, 'power': 4.3065e-05},
    (3.125, 16, 256) : {'area': 299.375990625, 'power': 1.19625e-05},
    (4.6875, 16, 256) : {'area': 299.375990625, 'power': 9.4468125e-06},
    (7.8125, 16, 256) : {'area': 299.375990625, 'power': 6.931125e-06},
    (14.0625, 16, 256) : {'area': 299.375990625, 'power': 4.415437499999999e-06},
    (26.5625, 16, 256) : {'area': 299.375990625, 'power': 1.89975e-06},
    
    (0.78125, 16, 512) : {'area': 321.460637, 'power': 4.57245e-05},
    (3.125, 16, 512) : {'area': 321.460637, 'power': 1.270125e-05},
    (4.6875, 16, 512) : {'area': 321.460637, 'power': 1.00305e-05},
    (7.8125, 16, 512) : {'area': 321.460637, 'power': 7.360125e-06},
    (14.0625, 16, 512) : {'area': 321.460637, 'power': 4.6895625e-06},
    (26.5625, 16, 512) : {'area': 321.460637, 'power': 2.019e-06}

}


# CLK_PERIOD, num_INPUT, BIT_INPUT, BIT_OUTPUT
Add_base = {

    # 正確版
#     (3.125,   3, 28, 29) : {"area": 332.261989, "power":  2.203e-04 },
#     (4.6875,  3, 28, 29) : {"area": 332.261989, "power": 1.5e-04},
#     (7.8125,  3, 28, 29) : {"area": 332.261989, "power": 1.2e-04},
#     (14.0625, 3, 28, 29) : {"area":  332.261989, "power": 8.0e-05},
#     (26.5625, 3, 28, 29) : {"area":  332.261989, "power": 2.915e-05 },
# 
#     (3.125,   16, 21, 28) : {"area": 1252.843168,  "power":  5.382e-04},
#     (4.6875,  16, 21, 28) : {"area": 1252.84316,  "power":  3.642e-04},
#     (7.8125,  16, 21, 28) : {"area": 1252.84316,  "power":  3.2e-04},
#     (14.0625, 16, 21, 28) : {"area": 1252.84316,  "power":  2.2e-04},
#     (26.5625, 16, 21, 28) : {"area": 1252.843168,  "power":  7.751e-05},
#     
#     (3.125,   16, 21, 29) : {"area": 1253.296768, "power":  5.383e-04},
#     (4.6875,  16, 21, 29) : {"area": 1253.296768, "power": 4.0e-04},
#     (7.8125,  16, 21, 29) : {"area": 1253.296768, "power": 3.5e-04},
#     (14.0625, 16, 21, 29) : {"area":  1253.296768, "power": 2.5e-04},
#     (26.5625, 16, 21, 29) : {"area": 1253.296768, "power":  7.752e-05}

    # *0.3 版
    (0.78125, 3, 28, 29) : {'area': 332.261989, 'power': 23.7924e-05},
    (3.125, 3, 28, 29) : {'area': 332.261989, 'power': 6.609e-05},
    (4.6875, 3, 28, 29) : {'area': 332.261989, 'power': 4.4999999999999996e-05},
    (7.8125, 3, 28, 29) : {'area': 332.261989, 'power': 3.6e-05},
    (14.0625, 3, 28, 29) : {'area': 332.261989, 'power': 2.4e-05},
    (26.5625, 3, 28, 29) : {'area': 332.261989, 'power': 8.745e-06},
    
    
    (0.78125, 2, 21, 28) :   {'area': 178.491593, 'power': 0.000145692}, # 要跟 16 input 的拿來內插
    (3.125, 2, 21, 28) :   {'area': 178.491593, 'power': 0.00004047}, # 要跟 16 input 的拿來內插
    (4.6875, 2, 21, 28) :  {'area': 178.491593, 'power': 0.000027162}, # 要跟 16 input 的拿來內插
    (7.8125, 2, 21, 28) :  {'area': 178.491593, 'power': 0.000016509}, # 要跟 16 input 的拿來內插
    (14.0625, 2, 21, 28) : {'area': 178.491593, 'power': 0.0000094}, # 要跟 16 input 的拿來內插
    (26.5625, 2, 21, 28) : {'area': 178.491593, 'power': 0.000005232}, # 要跟 16 input 的拿來內插

    

    (0.78125, 16, 21, 28) : {'area': 1252.843168, 'power': 0.000581256},
    (3.125, 16, 21, 28) : {'area': 1252.843168, 'power': 0.00016146},
    (4.6875, 16, 21, 28) : {'area': 1252.84316, 'power': 0.00010926000000000001},
    (7.8125, 16, 21, 28) : {'area': 1252.84316, 'power': 9.6e-05},
    (14.0625, 16, 21, 28) : {'area': 1252.84316, 'power': 6.6e-05},
    (26.5625, 16, 21, 28) : {'area': 1252.843168, 'power': 2.3253e-05},
    
    (0.78125, 16, 21, 29) : {'area': 1253.296768, 'power': 0.000581364},
    (3.125, 16, 21, 29) : {'area': 1253.296768, 'power': 0.00016149},
    (4.6875, 16, 21, 29) : {'area': 1253.296768, 'power': 0.00012},
    (7.8125, 16, 21, 29) : {'area': 1253.296768, 'power': 0.00010499999999999999},
    (14.0625, 16, 21, 29) : {'area': 1253.296768, 'power': 7.5e-05},
    (26.5625, 16, 21, 29) : {'area': 1253.296768, 'power': 2.3256e-05}


}


# # CLK_PERIOD, BIT_OUTPUT
# Add = {
#     (3.125,   29) : {"area": , "power": },
#     (4.6875,  29) : {"area": , "power": },
#     (7.8125,  29) : {"area": , "power": },
#     (14.0625, 29) : {"area": , "power": },
#     (26.5625, 29) : {"area": , "power": }
# 
# }
# 
# 
# # CLK_PERIOD, BIT_OUTPUT
# Accumulator ={
#     (3.125, 29) : {"area": , "power": },
#     (4.6875, 29) : {"area": , "power": },
#     (7.8125, 29) : {"area": , "power": },
#     (14.0625, 29) : {"area": , "power": },
#     (26.5625, 29) : {"area": , "power": }
# }


# CLK_PERIOD, POOLING_SIZE, BIT_IFM
Pooling ={
    (0.78125, 2, 8) : {"area":  306.46349, "power": 24.5295e-05},
    (3.125, 2, 8) : {"area":  306.46349, "power": 6.81375e-05},
    (4.6875, 2, 8) : {"area": 306.97379, "power": 4.0e-05},
    (7.8125, 2, 8) : {"area": 306.97379, "power": 3.5e-05},
    (14.0625, 2, 8) : {"area":306.97379 , "power": 2.5e-05},
    (26.5625, 2, 8) : {"area": 306.97379, "power": 9.455e-06}
}



# CLK_PERIOD, WIRE_NUM
Router ={
    (0.78125, 29) : {"area": 2859.947888, "power": 19.3572e-04},
    (3.125, 29) : {"area": 2859.947888, "power": 5.377e-04},
    (4.6875, 29) : {"area": 2859.721088, "power": 3.0e-04},
    (7.8125, 29) : {"area": 2859.721088, "power": 2.1e-04},
    (14.0625, 29) : {"area":2859.721088 , "power": 9.0e-05},
    (26.5625, 29) : {"area":  2859.721088, "power": 7.916e-05}
}
