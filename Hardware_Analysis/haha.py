import sys
sys.path.append('../..')
# from MyWork import Config
# from ..MyWork import Config
# import Area_Model
import numpy as np
# from MyWork import Config_copy
# import Area_Model
import pickle
from itertools import combinations_with_replacement
import math
import scipy

class ClassA():
    def __init__(self,a):
        self.a = a
        print(f"self.a = {self.a}")
    def geta(self):
        print(f"self.a = {self.a}")

class ClassB():
    def __init__(self,objectB):
        objectB.a = 10
        self.a = objectB.a
        self.a = 30
        self.lis = [1,2,3,4,5]
        self.lisB = None
    def get_list(self):
        print(f"list is {self.lis}")
        print(f"listB is {self.lisB}")

a=1

def change_list(object):
    haha = object.lis
    haha.append(6)
    return haha

def f(b):
    b=b+1
    return b

def f_list(a):
    # a[0]=2
    a.append(4)
    b=a.copy() 
    b=a
    b.append(5)
    return a

def empty_list(k):
    # print(f"emptyli is {emptyli}")
    emptyli=[]
    for i in range(k):
        emptyli.append(k)
    return emptyli

def funa(a):
    a = a - 3

    return a

def funb(a):
    for i in range(len(a)):
        a[i] = a[i]-3
    return a

def my_func(a):
    b=a
    a[1] = 2.0
    return b

def change_numpy(a):
    a[0] = 13

def sort_the_key_of_each_cluster(dictionary_list):
    for i, ei in enumerate(dictionary_list):
        dictionary_list[i] = dict(sorted(dictionary_list[i].items())) 

def create_OU_table(dictionary_list, OU):

    #
    sort_the_key_of_each_cluster(dictionary_list)


    #
    OU_table = dict()

    for i, ei in enumerate(dictionary_list):
        flattened_dic = flatten_dictionary(dictionary_list[i])
        append_to_OU_table(i, flattened_dic, OU, OU_table)

    return OU_table


    
def flatten_dictionary(dic):
    """ Flatten the dictionary into a list of [key, value] pairs. """
    return [[key, value] for key, values in dic.items() for value in values]

def append_to_OU_table(cluster_idx, flattened, OU, OU_table):
    print(f"flattened = {flattened}")
    for i in range(0, len(flattened), OU):
        # Extract the current chunk
        chunk = flattened[i:i + OU]

        # Create the key for OU_table (e.g., 'aa', 'ab', etc.)
        ou_key = ''.join(item[0] for item in chunk)
        
        # Prepare the values to append
        values_to_append = [[item[1]] for item in chunk]
        print(f"values to appennd = {values_to_append}")

        # Append to OU_table
        if ou_key in OU_table:
            OU_table[ou_key].append((cluster_idx, values_to_append))
        else:
            OU_table[ou_key] = [(cluster_idx, values_to_append)]

    


# print(f"Config is {Config.CLK_PERIOD}")
if __name__ == '__main__':
    __A = ClassA(20)
    __B = ClassB(__A)
    __A.geta()
    __B.lisB = change_list(__B)
    __B.get_list()

    # a=f(a)
    # print(a)

    a=[1,2,3]
    a=f_list(a)
    print(a)

    #
    li = [[] for i in range(3)]
    for i in range(3):
        print(f"is is {i}")
        # li[i] = empty_list(li[i],i+1)
        li[i] = empty_list(i+1)
    print(f"li is {li}")
    li_multiply = [[] for i in range(3)]
    for i in range(3):
        li_multiply[i] = [2*j for j in li[i]]
    print(f"li is {li}")
    print(f"li_multiply is {li_multiply}")

    #
    arra = np.array([3, 4, 5])
    resa = funa(arra)
    print(f"arra = {arra}")
    print(f"funa = {resa}")


    #
    arrb = np.array([3, 4, 5])
    resb = funb(arrb)
    print(f"arrb = {arrb}")
    print(f"funb = {resb}")


    #
    func_resb = my_func(arrb)
    print(f"arr = {arrb}")
    print(f"func_res = {func_resb}")


    # 
    before_slice = [1,2,3,4,5,6,7,8]
    after_slice = before_slice[4:6]
    after_slice = [0,1]
    print(f"before slice = {before_slice}")
    print(f"after slice = {after_slice}")



    #
    arr = [1,2,3,4,5,6,7]
    idx = [2,3]
    for i in idx:
        arr[i] = arr[i]+1
    print(f"arr = {arr}")

    #
    total_layer_num = 4
    # zero_array =  total_layer_num * [0]
    zero_array = [0 for i in range(total_layer_num)]
    print(f"zero_array = {zero_array}")


    #
    area_string = "area"
    # print(f"Config_copy.Activation[(1,2,3)][area] = {Config_copy.Activation[(1,2,3)][area_string]}")
    # print(f"Config_copy.Activation[(1,2,4)][area] = {Config_copy.Activation[(1,2,4)][area_string]}")


    # ADC
    area = 1200
    power = 0.002
    BIT = [1,2,3,4,5,6,7,8]
    for i in BIT:
        print(f"({i}, 1.28): ")
        print(f"    power = {power * 2**(i-8) }") 
        print(f"    area = {600*2**(i-8) + 600*(i/8) }") 

    # scaling
    max =  6.478e-04
    min = 1.488e-04
    differcne = max - min
    for i in range(1,4):
        print(f"{i} : {min + differcne*i/4}")


    #
    list_find = [1,2,1,4,1]
    list_find_idx = list_find.index(1)
    print(f"list_find_idx = {list_find_idx}")

    # test numpy array mutable or not
    numpy_array = np.array([1, 2, 3, 4])
    change_numpy(numpy_array)
    print(f"numpy_array after change is {numpy_array}")


    # test if sort indices can change original array
    numpy_array = np.array([1, 2, 3, 4])
    sorted_indices = [3,2,1,0]
    numpy_array_after_sort = numpy_array[sorted_indices]
    print(f"numpy_array_after_sort = {numpy_array_after_sort}")
    numpy_array_after_sort[0] = 5
    print(f"numpy_array_after_sort after numpy_array_after_sort is changed = {numpy_array_after_sort}")
    print(f"numpy_array_after      afer  numpy_array_after_sort is changed = {numpy_array}")



    # dictionary list 轉成 OU_table
    dica = {
        'c': ['v9', 'v10', 'v13', 'v18'],
        'a': ['v4', 'v5', 'v6', 'v7', 'v8'],
        'b': ['v1', 'v2', 'v3']
    }

    dicb = {
        'c': ['v13', 'v28', 'v39', 'v66'],
        'a': ['v4', 'v5', 'v6', 'v7', 'v8'],
        'b': ['v0', 'v2', 'v4', 'v5', 'v9']
    }
    OU = 2
    dictionary_list = []
    dictionary_list.append(dica)
    dictionary_list.append(dicb)

    OU_table = create_OU_table(dictionary_list, OU)
    print(f"dictionnary_list = {dictionary_list}")
    print(f"OU_talbe = {OU_table}")




    # baseline ADC energy
    # OFM_row     = [224,224,112,112,56,56,56,28,28,28,14,14,14]
    OFM_row     = [32,32,16,16,8,8,8,4,4,4,2,2,2]
    IN_CH       = [3,64,64,128,128,256,256,256,512,512,512,512,512]
    OUT_CH      = [64,64,128,128,256,256,256,512,512,512,512,512,512]
    K=3
    BIT_W = 8
    BIT_IFM = 8
    BIT_PER_CELL = 2
    
    # OU = 2
    # ADC_read_energy = 9.375e-05 * 3.125

    
    # OU = 4
    # ADC_read_energy = 0.00014062500000000002  * 4.6875
    
    # OU = 8
    # ADC_read_energy = 0.00022500000000000002 * 7.8125
    
    # OU = 16
    # ADC_read_energy = 0.000375 * 14.0625
    
    # OU = 32
    # ADC_read_energy = 0.0006428571428571429 * 26.5625

    OU = 128
    ADC_read_energy = 0.02245
    
    
    # (Config.NETWORK_DICT["K"][i] \
    #     * Config.NETWORK_DICT["BIT_W"] * Config.NETWORK_DICT["OUT_CH"][i] \
    #     * Config.NETWORK_DICT["IN_CH"][i]) \
    #     * 0.1 / (Config.num_Macro_per_Tile * mywork.OU**2) for i in range(Config.NETWORK_DICT["total_layer_num"])]
    

    OU_cycle_of_each_layer = [0 for i in range(13)]
    ADC_each_layer_switch_cycle = [0 for i in range(13)]
    ADC_each_layer_energy = [0 for i in range(13)]
    for i in range(13):
        OU_cycle_of_each_layer[i] = OFM_row[i]**2 * BIT_IFM * (BIT_W * (1/BIT_PER_CELL) * OUT_CH[i] / OU) * (K**2 * IN_CH[i] / OU)
        ADC_each_layer_switch_cycle[i]  = OU_cycle_of_each_layer[i] * OU
        ADC_each_layer_energy[i] = ADC_each_layer_switch_cycle[i] * ADC_read_energy
    print(f"OU_cycle_of_each_layer  = {OU_cycle_of_each_layer }")    
    print(f"ADC_each_layer_switch_cycle = {ADC_each_layer_switch_cycle}")
    print(f"ADC_each_layer_energy = {ADC_each_layer_energy}nJ")
    print(f"ADC_total_energy = {sum(ADC_each_layer_energy)/1e+06}mJ")
    

    num_Macro_of_each_layer = [0 for i in range(13)]
    for i in range(13):
        num_Macro_of_each_layer[i] = (BIT_W * (1/BIT_PER_CELL) * OUT_CH[i] / 128) * (K**2 * IN_CH[i] / 128)
    print(f"num_Macro_of_each_layer = {num_Macro_of_each_layer}")
    print(f"sum of num_Macro_of_each_layer = {sum(num_Macro_of_each_layer)}")


    # sum of sum of list
    # two_dimension_list = [[1,2,3],[4,5,6]]
    # print(f"sum of two_dimension_list = {sum(sum(two_dimension_list))}")

    # Macro num
    PE_num_Macro = [[1, 1, 1], [10, 9, 10], [10, 10, 10], [19, 19, 20], [19, 19, 19], [38, 38, 38], [38, 37, 38], [38, 38, 37], [75, 74, 74], [74, 75, 74], [38, 38, 38], [32, 37, 33], [30, 37, 31]]
    print(f"macro num = {sum([sum(haha) for haha in PE_num_Macro])}")

    # PE_cycle for each OU shape
    # OU=4
    # num_Macro_per_Tile = 64
    # PE_idx = 0
    # for layer_idx in range(13):
    #     file_name = "/home/mark/MyWork_OU4/PE_OU_cycle_for_each_OU_shape/" \
    #                 + "CONV" + str(layer_idx+1) \
    #                 + '/OU=' + str(OU) \
    #                 + '_num_Macro_per_Tile=' + str(num_Macro_per_Tile)\
    #                 + '_PE' + str(PE_idx) \
    #                 + '.pickle'
    # with open(file_name, 'rb') as file:
    #     PE_OU_cycle_for_each_OU_shape = pickle.load(file)
    # print(f"PE_OU_cycle_for_each_OU_shape = {PE_OU_cycle_for_each_OU_shape}")


    # OU shape combination
    # OU=4
    # print(f"num OU shape = { math.comb(2**OU-1+OU-1,OU)}")
    # print(f"num OU per Macro = {(128/OU)**2}")


    # power 修改估計版
    Adder_mask = {
        (3.125, 512,   27) : {'area':  11186.455908, 'power': 6.478e-04},
        (4.6875, 512,  27) : {'area': 11185.548708, 'power': 0.0005230500000000001},
        (7.8125, 512,  27) : {'area': 11185.548708, 'power': 0.0003983},
        (14.0625, 512, 27) : {'area': 11185.548708, 'power': 0.00027355},
        (26.5625, 512, 27) : {'area': 11185.548708, 'power': 1.488e-04}



        }
    for key, value in Adder_mask.items():
        value["power"] = value["power"]*0.3
        value["area"] = value["area"]
        print(f"{key} : {value},")



    # num_operations 公式計算
    out_ch_list = [3,64,64,128,128,256,256,256,512,512,512,512,512]
    in_ch_list = [64,64,128,128,256,256,256,512,512,512,512,512,512]
    k_list = [ 3, 3,3,3,3,3,3,3,3,3,3,3,3]
    ofm_row_list = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    num_operations_of_each_layer = [out_ch_list[i] * in_ch_list[i] * k_list[i]**2 \
        * ofm_row_list[i]**2 * 2 for i in range(13)]
    num_operations = sum( num_operations_of_each_layer )
    print(f"num_operations of each layer = {num_operations_of_each_layer}")
    print(f"sum of num_operations_of_each_layer = {sum(num_operations_of_each_layer)/1e+09}GOPs")


    # ou power
    device_power = 2.2020833333333332e-07
    OU_size = [2,4,8,16,32,128]
    for i in OU_size:
        print(f"OU_size = {i}, power = {device_power*(i**2)}")


    # 除法
    a = 13 / 2 / 4
    print(f"13/2/4 = {a}")



    # power scaling

    Row_Buffers = {

        (3.125, 448, 12) : {"area":49764.980762 , "power": 7.162e-03 },
        (4.6875, 448, 12) : {"area": 49764.527162 , "power": 4.891e-03},
        (7.8125, 448, 12) : {"area":49766.795162 , "power": 3.074e-03},
        (14.0625, 448, 12) : {"area": 49766.795162 , "power": 1.863e-03},
        (26.5625, 448, 12) : {"area":49766.795162 , "power": 1.150e-03},


        (3.125, 1792, 32) : {"area": 157141.425963 , "power":  0.0157},
        (4.6875, 1792, 32) : {"area":157142.333163 , "power":  0.0109},
        (7.8125, 1792, 32) : {"area":157142.333163 , "power": 7.022e-03},
        (14.0625, 1792, 32) : {"area":157152.539164 , "power": 4.448e-03},
        (26.5625, 1792, 32) : {"area":157153.219564 , "power": 2.935e-03 },


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

    Shift_and_Add = {
        (3.125, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 1.74e-05}, 
        (4.6875, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 1.178e-05}, 
        (7.8125, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 7.284e-06}, 
        (14.0625, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 4.288e-06}, 
        (26.5625, 1, 1, 2, 8, 8): {'area': 402.569984, 'power': 2.526e-06}, 
        
        (3.125, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 1.9442e-05}, 
        (4.6875, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 1.3186e-05}, 
        (7.8125, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 8.182e-06}, 
        (14.0625, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 4.846000000000001e-06}, 
        (26.5625, 2, 1, 2, 8, 8): {'area': 474.238783, 'power': 2.884e-06}, 
        
        (3.125, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 1.9442e-05}, 
        (4.6875, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 1.3186e-05}, 
        (7.8125, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 8.182e-06}, 
        (14.0625, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 4.846000000000001e-06}, 
        (26.5625, 1, 1, 4, 8, 8): {'area': 474.238783, 'power': 2.884e-06}, 
        
        (3.125, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 2.1340000000000002e-05}, 
        (4.6875, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 1.4496e-05}, 
        (7.8125, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 9.02e-06}, 
        (14.0625, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 5.368e-06}, 
        (26.5625, 2, 1, 4, 8, 8): {'area': 547.26838, 'power': 3.2199999999999997e-06}, 
        
        (3.125, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 2.1340000000000002e-05}, 
        (4.6875, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 1.4496e-05}, 
        (7.8125, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 9.02e-06}, 
        (14.0625, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 5.368e-06}, 
        (26.5625, 1, 1, 8, 8, 8): {'area': 547.26838, 'power': 3.2199999999999997e-06}, 
        
        (3.125, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 2.3420000000000003e-05}, 
        (4.6875, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 1.846e-05}, 
        (7.8125, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 1.3500000000000001e-05}, 
        (14.0625, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 8.540000000000001e-06}, 
        (26.5625, 2, 1, 8, 8, 8): {'area': 624.153578, 'power': 3.5800000000000005e-06}, 
        
        (3.125, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 2.3420000000000003e-05}, 
        (4.6875, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 1.846e-05}, 
        (7.8125, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 1.3500000000000001e-05}, 
        (14.0625, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 8.540000000000001e-06}, 
        (26.5625, 1, 1, 16, 8, 8): {'area': 624.153578, 'power': 3.5800000000000005e-06}, 
        
        (3.125, 2, 1, 16, 8, 8): {'area': 692.420375, 'power': 2.556e-05}, 
        (4.6875, 2, 1, 16, 8, 8): {'area': 692.42037, 'power': 2.0158e-05},
        (7.8125, 2, 1, 16, 8, 8): {'area': 692.42037, 'power': 1.4755999999999999e-05}, 
        (14.0625, 2, 1, 16, 8, 8): {'area': 692.42037, 'power': 9.354e-06}, 
        (26.5625, 2, 1, 16, 8, 8): {'area': 692.420375, 'power': 3.9520000000000004e-06}, 
        
        (3.125, 1, 1, 32, 8, 8): {'area': 692.420375, 'power': 2.556e-05}, 
        (4.6875, 1, 1, 32, 8, 8): {'area': 692.42037, 'power': 2.0158e-05}, 
        (7.8125, 1, 1, 32, 8, 8): {'area': 692.42037, 'power': 1.4755999999999999e-05}, 
        (14.0625, 1, 1, 32, 8, 8): {'area': 692.42037, 'power': 9.354e-06}, 
        (26.5625, 1, 1, 32, 8, 8): {'area': 692.420375, 'power': 3.9520000000000004e-06}, 
        
        (3.125, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 2.6400000000000005e-05}, 
        (4.6875, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 2.08065e-05}, 
        (7.8125, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 1.5213e-05}, 
        (14.0625, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 9.619500000000002e-06}, 
        (26.5625, 2, 1, 32, 8, 8): {'area': 700.0836, 'power': 4.026e-06}
    }

    Mux_base = {
        (3.125, 8, 21)   : {'area': 229.294789, 'power': 0.0000417},
        (4.6875, 8, 21)  : {'area': 229.294789, 'power': 0.000027969},
        (7.8125, 8, 21)  : {'area': 229.294789, 'power': 0.000016977},
        (14.0625, 8, 21) : {'area': 229.294789, 'power': 0.000009651},
        (26.5625, 8, 21) : {'area': 229.294789, 'power': 0.00000534}, 


        (3.125, 32, 32)    : {'area': 1006.538354, 'power': 0.0001287},
        (4.6875, 32, 32)   : {'area': 1005.63115, 'power': 0.00010076999999999999},
        (7.8125, 32, 32)   : {'area': 1005.63115, 'power': 7.284e-05},
        (14.0625, 32, 32)  : {'area': 1005.63115, 'power': 4.491e-05},
        (26.5625, 32, 32)  : {'area': 1005.631154, 'power': 1.698e-05},


        (3.125, 3, 20)     : {'area': 130.863595, 'power': 3.237e-05},
        (4.6875, 3, 20)    : {'area': 130.863595, 'power': 2.5289249999999995e-05},
        (7.8125, 3, 20)    : {'area': 130.863595, 'power': 1.82085e-05},
        (14.0625, 3, 20)   : {'area': 130.863595, 'power': 1.1127749999999999e-05},
        (26.5625, 3, 20)   : {'area': 130.863595, 'power': 4.0469999999999995e-06},
        
        (3.125, 64, 20) : {'area': 1144.659548, 'power': 6.867e-05},
        (4.6875, 64, 20) : {'area': 1144.659548, 'power': 5.4071999999999986e-05},
        (7.8125, 64, 20) : {'area': 1144.659548, 'power': 3.9474e-05},
        (14.0625, 64, 20) : {'area': 1144.659548, 'power': 2.4876e-05},
        (26.5625, 64, 20) : {'area': 1144.886348, 'power': 1.0278e-05},


        (3.125, 64, 25)    : {'area': 1449.705543, 'power': 7.026e-05},
        (4.6875, 64, 25)   : {'area': 1449.705543, 'power': 5.5475999999999995e-05},
        (7.8125, 64, 25)   : {'area': 1449.705543, 'power': 4.0691999999999996e-05},
        (14.0625, 64, 25)  : {'area': 1449.705543, 'power': 2.5908e-05},
        (26.5625, 64, 25)  : {'area': 1449.705543, 'power': 1.1123999999999999e-05},
        
        (3.125, 128, 25) : {'area': 2737.47589, 'power': 0.00010017},
        (4.6875, 128, 25) : {'area': 2737.47589, 'power': 7.940174999999999e-05},
        (7.8125, 128, 25) : {'area': 2737.47589, 'power': 5.86335e-05},
        (14.0625, 128, 25) : {'area': 2737.47589, 'power': 3.78651e-05},
        (26.5625, 128, 25) : {'area': 2737.47589, 'power': 1.7097e-05},
        
        (3.125, 128, 26)   : {'area': 2844.752286, 'power': 0.00010647},
        (4.6875, 128, 26)  : {'area': 2844.752286, 'power': 8.436675e-05},
        (7.8125, 128, 26)  : {'area': 2844.752286, 'power': 6.22635e-05},
        (14.0625, 128, 26) : {'area': 2844.752286, 'power': 4.016025e-05},
        (26.5625, 128, 26) : {'area': 2844.752286, 'power': 1.8057e-05},
        
        (3.125, 256, 26) : {'area': 5476.312561, 'power': 6.644999999999999e-05},
        (4.6875, 256, 26) : {'area': 5475.858961, 'power': 5.591699999999999e-05},
        (7.8125, 256, 26) : {'area': 5475.858961, 'power': 4.5384e-05},
        (14.0625, 256, 26) : {'area': 5475.858961, 'power': 3.4850999999999995e-05},
        (26.5625, 256, 26) : {'area': 5475.858961, 'power': 2.4318e-05},
        
        (3.125, 256, 27)   : {'area': 5693.813754, 'power': 0.00012146999999999999},
        (4.6875, 256, 27)  : {'area': 5695.628154, 'power': 9.743325e-05},
        (7.8125, 256, 27)  : {'area': 5695.628154, 'power': 7.33965e-05},
        (14.0625, 256, 27) : {'area': 5695.628154, 'power': 4.9359750000000005e-05},
        (26.5625, 256, 27) : {'area': 5695.628154, 'power': 2.5322999999999998e-05},
        
        (3.125, 512, 27) : {'area': 11186.455908, 'power': 0.00019434},
        (4.6875, 512, 27) : {'area': 11185.548708, 'power': 0.000156915},
        (7.8125, 512, 27) : {'area': 11185.548708, 'power': 0.00011948999999999998},
        (14.0625, 512, 27) : {'area': 11185.548708, 'power': 8.2065e-05},
        (26.5625, 512, 27) : {'area': 11185.548708, 'power': 4.464e-05},
        
        
        (3.125, 512, 28)   : {'area': 11586.757889, 'power': 0.00020325},
        (4.6875, 512, 28)  : {'area': 11589.93309, 'power': 0.0001640475},
        (7.8125, 512, 28)  : {'area': 11589.93309, 'power': 0.000124845},
        (14.0625, 512, 28) : {'area': 11589.93309, 'power': 8.56425e-05},
        (26.5625, 512, 28) : {'area': 11589.93309, 'power': 4.6439999999999996e-05}
    }
    rb_list = []
    clk_period = [0.78125, 3.125, 4.6875, 7.8125, 14.0625, 26.5625]
    for key, value in Row_Buffers.items():
        if( key[0] == 3.125 ):
            temp_tuple = (float(value["area"]), float(value["power"])*3.6)
            rb_list.append(temp_tuple)

        temp_tuple = (float(value["area"]), float(value["power"]))
        rb_list.append(temp_tuple)
    

    area_string = "area"
    power_string = "power"

    for i in range(6):
        print(f"clk_period = {clk_period[i] :>15f}, \
            area = { deco_list[i][0] + mux_of_SA_list[i][0] + add_list[i][0] + mux_of_filter_list[i][0] :>15f}, \
            power = {deco_list[i][1] + mux_of_SA_list[i][1] + add_list[i][1] + mux_of_filter_list[i][1] :>15f}\
            ")

# 
# 前面 group_after_CoarseFine 
# 已經有考慮 bit 了
# 
# 
# # 原本
# list_A = [0, 2], bit_per_cell=2, list_C = [0,1,4,5]
# dictionary_list = []
# shape_union = set()
# for iterate  cluster_contain_what_indices:
#     
#     d  裝 bitline shape 用的
#     for cluster 內每條 bitline:
#         轉換成 decimal_number
#         d[decimal_number].append(bitline 位置) ..... (式1)
#         shape_union.add(decimal_number)
#         ex. 01100110(垂直的) 是 column 0, 2
#             11100110(垂直的) 是 column 3,4,6,7
#     
#     dictionary_list.append(d)
# 
# 
# OU_table = dict()
# for iterate dictionary_list:
#     sort dictionary 
#     ex. shape 從 1,2,3, 排序
# 
#     for sort 完的 dictionary:
#         OU_tuple = 一次抓 OU 個 shape # (shapea, shapeb)
#         output_vectors = [[shapea 有哪些 output vectors], [shapeb 有哪些 output vectors]]
#         OU_table[OU_tuple].append( 第幾 cluster,  output_vectors)
# 
# 
# aaaaaaabbbbbbb => ab
# cluster1_d[a] = [0,1,2,3,4,5]
# cluster1_d[b] = [6,7,8,9,10,11]
# ....c
# ....d
# 
# 
# # 改成 bitline 取消
# bitline_shape_dictionary_of_every_cluster = []
# for iterate  cluster_contain_what_indices:
#     for cluster 內每條 bitline:
#         轉換成 decimal_number
#         bitline_shape_dictionary[decimal_number].append(bitline 位置)
#     bitline_shape_dictionary_of_every_cluster.append(bitline_shape_dictionary)
# 
# 
# cluster 1
# a : v4, v5, v6, v7..... 共 19 個
# b : ..... 共 23 個
# c : ..... 共 19 個
# d : ..... 共 23 個
# 
# cluster 2
# a : 9  個
# b : 13 個
# c : 12 個
# e : 23 個
# 
# 
# OU_table = dict()
# for iterate dictionary_list:
#     sort dictionary 
#     ex. shape 從 1,2,3, 排序
#     
#     舊
#     for sort 完的 dictionary:
#         OU_tuple = 一次抓 OU 個 shape # (shapea, shapeb)
#         output_vectors = [[shapea 有哪些 output vectors], [shapeb 有哪些 output vectors]]
#         OU_table[OU_tuple].append( 第幾 cluster,  output_vectors)
# 
#     新
#     for sort 完的 dictionary:
#         一次抓 OU 個 shape => 改成 => 一次抓 OU 個 vector
# 
#         (key, value)
#             (c, [v9,v10,v13,v18....])
#             (a, [v4, v5, v6, v7.....])
#             (b, [v1,v2,v3...])
#             
#         展開成
#         
# 
#         
#         OU_table[aa].append( (cluster1, [[v4],[v5]]) )
#         OU_table[aa].append( (cluster1, [[v6],[v7]]) )
#         ....
#         OU_table[ab].append( (cluster1, [[v6],[v7]]) )
# 
# suppose i have a parameter called OU and a dictionary called dic and looks like
# (key, value)
#     (c, [v9,v10,v13,v18....])
#     (a, [v4, v5, v6, v7, v8])
#     (b, [v1,v2,v3...])
# now i want to first sort the the key 
# (key, value)
#     (a, [v4, v5, v6, v7, v8])
#     (b, [v1,v2,v3...])
#     (c, [v9,v10,v13,v18....])
# then according to OU meaning every time i want to take how many  
# element in side the key(which is a list), then append to a dictionary
# so suppose OU=2, it would look like
# OU_table[aa].append( [[v4],[v5]] )
# OU_table[aa].append( [[v6],[v7]] )
# OU_table[ab].append( [[v8],[v1]] )
# OU_table[bb].append( [[v2],[v3]] )
# .... 
# keep going 
# 
# 
# 
# lets just say 
# 1. first we want to
# flatten the dictionary 
# (key, value)
#     (a, [v4, v5, v6, v7, v8])
#     (b, [v1,v2,v3...])
#     (c, [v9,v10,v13,v18....])
# into
# [a,[v4]], [a,[v5]], [a,[v6]]....[b,[v1]].....
# then 
# 2.
# OU_table[aa].append( [[v4],[v5]] )
# OU_table[aa].append( [[v6],[v7]] )
# OU_table[ab].append( [[v8],[v1]] )
# OU_table[bb].append( [[v2],[v3]] )
# 
# idx 4     7     13    19     
#     01    01    11    10    
#     10    10    11    01    
#     01    01    01    11    
#     11    11    10    00
# 
# 
# cluster 1
# a : 19 個
# b : 23 個
# c : 19 個
# d : 23 個
# 
# cluster 2
# a : 9  個
# b : 13 個
# c : 12 個
# e : 23 個
# 
# shape 有 2^OU -1 種
# OU=2 => 3
# OU=4 => 15
# OU=8 => 63
# OU=16 => 很多
# OU=32 => 很多
# 
# 
# a//OU + b//OU + c//OU + d//OU
# (a%OU  + b%OU + c%OU + d%OU)/OU
# 
# 
# 
# aa :    (cluster1, 9)
#         (cluster2, 4)
# ab :    ()
# 
# 
# parallelize_OU_to_Macro(
#     cycle_for_each_OU=ej, 
#     num_Macro=need_num_Macro[i], 
#     num_OU_per_Macro=num_OU_per_Macro)

