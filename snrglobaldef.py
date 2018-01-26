import os
from  nets.spncnn_model import SPNCNNModel
    

g_model_class_name = SPNCNNModel  

g_modellogdir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
g_modellogdir = "{}/logs/{}".format(g_modellogdir, g_model_class_name.__name__)




