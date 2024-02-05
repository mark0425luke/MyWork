
import math

def SRAM_size():
    # 要 STRIDE*K - STRIDE + K 個 SRAM
    # SRAM 方正 : mux_width * sram_width = #word / mux_width => mux^2 = #word / sram_width 
    # Artisan Memory Compiler 的 mux_width 有 8, 16, 32
    # 所以當挑 mux_width=8 時 #word / sram_width = 64
    # 又 weight matrix 是要存 OFM_row 個 (IN_CH*BIT_IFM)
    # 所以我就訂 sram_width, 然後再得到 #word = OFM_row * IN_CH * BIT_IFM / sram_width
    # 所以 #word / sram_width = (OFM_row*IN_CH*BIT_IFM) / (sram_width**2) = 64
    # 代表 sram_width 要開 大概 (OFM_row*IN_CH*BIT_IFM)**(0.5) / 8

    # 參數設定
    OFM_row = [224,224,112,112,56,56,56,28,28,28,14,14,14]
    # OFM_row = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    IN_CH   = [3,64,64,128,128,256,256,256,512,512,512,512,512,512]
    BIT_IFM = 8
    your_sram_width = [12, 48, 32, 48, 32, 48, 48, 32, 48, 48, 32, 32, 32]
    # your_sram_width = [8, 16, 8, 16, 8, 16, 16, 8, 16, 16, 8, 8, 8]
    # your_sram_width = [4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    mux_width = 8

    # ideal
    ideal_sram_width = [(OFM_row[i]*IN_CH[i]*BIT_IFM / mux_width**2)**(0.5) for i in range(13)]

    # real
    sram_num_word = [int(math.ceil((OFM_row[i]*IN_CH[i]*BIT_IFM / your_sram_width[i]))) for i in range(13)]
    ratio = [(sram_num_word[i] / your_sram_width[i]) for i in range(13)]

    print(f"ideal sram width")
    print(f"     {ideal_sram_width}")
    print(f"your sram width ")
    print(f"     {your_sram_width}")
    print(f"sram_num_word")
    print(f"     {sram_num_word}")
    print(f" ratio of sram_num_word / sram_width")
    print(f"     {ratio}")

if __name__ == '__main__':
    
    SRAM_size()