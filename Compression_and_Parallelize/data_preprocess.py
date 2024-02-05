import Config
import numpy as np
import math
def multibit_per_cell(output_reshape, bit_per_cell):

    y = np.zeros(( output_reshape.shape[0],  int(math.ceil(output_reshape.shape[1]/bit_per_cell)) ), dtype=int)
    for i,ei in enumerate(output_reshape):
        chunks = np.split(ei, len(ei) // bit_per_cell) # ex. 11001010 切成 11, 00, 10, 10
        for j, chunk in enumerate(chunks):
            y[i][j] = 0 if np.all(chunk == 0) else 1 # ex. 11=>1, 00=>0, 10=>1, 10=>1

    # print(f"y shape = {y.shape}")
    return y


def padding_due_to_OUX(output_reshape, OUX): # 這邊是已經切割成 FullReuse 且經過 eliminate column 的 transpose，且 shape[0]=9，ex. CONV1 PE0 9x512, ex. OUX=16, 9x512 -> 16x512，
    
    
    if ((output_reshape.shape[0]%OUX) != 0) :
        # logging.info("need to do padding")
        padding = np.zeros((1,output_reshape.shape[1]),dtype=int)
        num_padding = OUX*math.ceil(output_reshape.shape[0]/OUX) - output_reshape.shape[0]
        print('num_paddings',num_padding)
        print("before padding ",output_reshape.shape)
        for j in range(num_padding):
            output_reshape = np.append(output_reshape, padding,axis=0) 
        print("after padding",output_reshape.shape)

    else:
        print("No Padding")
        print("output_reshape.shape : ",output_reshape.shape)


    return output_reshape


def load_binary_weight(layer_idx):
    home_dir = "/home/mark/k-means/"
    conv_prefix = "CONV" + str(layer_idx+1)
    size_prefix = \
        str(Config.NETWORK_DICT["OUT_CH"][layer_idx] * Config.NETWORK_DICT["BIT_W"]) \
        + 'x' + str(Config.NETWORK_DICT["IN_CH"][layer_idx]*Config.NETWORK_DICT["K"][layer_idx]*Config.NETWORK_DICT["K"][layer_idx])
    extension = "npy"
    file_name = home_dir + conv_prefix + '/' + conv_prefix + '_' + size_prefix + '.' + extension
    output_reshape = np.load(file_name)
    return output_reshape

def get_indices_for_FullReuse(k, output_reshape):
  # Create the indices as per the new specified pattern
  indices_list = []

  # Calculate the number of sets of 3 columns
  num_sets = output_reshape.shape[1] // (k*k)
  # print(f"num_sets : {num_sets}")

  for i in range(k):
    indices = []
    for j in range(num_sets):
        start = k*j*k + 3*i
        end = start + k
        indices.extend(range(start, end))
    indices_list.append(indices)

  return indices_list
  
def divide_weight_for_PE(k, output_reshape):
    # 因為 output_reshape 順序是 in_ch0 的 9 宮格，然後 in_ch1 的 9 宮格 ....
    # 所以要改成 PE 0 拿 0, 1, 2, 9, 10, 11
    #           PE 1 拿 3, 4, 5, 12, 13, 14.....
    FullReuse_indices = get_indices_for_FullReuse(k, output_reshape)
    

    # 
    weight_for_PE = np.zeros((k, output_reshape.shape[0], int(output_reshape.shape[1]/k)), dtype=int)
    for i, FullReuse_index in enumerate(FullReuse_indices):
        weight_for_PE[i] = output_reshape[:, FullReuse_index] # : 代表全部，即每個 filter 都取 FullReuse_index

    return weight_for_PE





def data_preprocess(layer_idx):

    # load binary weight, shape 是 (OUT_CH * BIT_W, IN_CH * K * K)
    output_reshape = load_binary_weight(layer_idx)

    # 切割給 k 塊 PE, weight_for_PE 的 shape 是 (k, OUT_CH * BIT_W, IN_CH*K)
    weight_for_PE = divide_weight_for_PE(Config.NETWORK_DICT["K"][layer_idx], output_reshape)

    
    return weight_for_PE