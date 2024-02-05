# import sys
# sys.path.append('..')

# import Area_Model
# import Energy_Model
from Hardware_Analysis import Area_Model
from Hardware_Analysis import Energy_Model
import Config



def Hardware_Analysis(mywork):
    

    # convolution
    mywork.conv_area_model      = Area_Model.CONV_Area_Model(mywork)
    mywork.conv_energy_model    = Energy_Model.CONV_Energy_Model(mywork)

    # Pooling
    mywork.pooling_area_Model   = Area_Model.Pooling_Area_Model(mywork)
    mywork.pooling_energy_Model = Energy_Model.Pooling_Energy_Model(mywork)
    


    # Router
    mywork.router_area_model = Area_Model.Router_Area_Model(mywork)
    mywork.router_energy_model = Energy_Model.Router_Energy_Model(mywork)


    return 

if __name__ == '__main__':
    print(f" main Hardware_Analysis, not doing things")