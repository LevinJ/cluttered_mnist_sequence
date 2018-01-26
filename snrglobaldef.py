import os
from  nets.spncnn_model import SPNCNNModel
    
class SnrModelType(object):
    spncnn = "spncnn"
    rnnspncnn = "rnnspncnn"
  
   
g_snrmodeltype = SnrModelType.spncnn

g_modellogdir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
g_modellogdir = "{}/logs/{}".format(g_modellogdir, g_snrmodeltype)

if g_snrmodeltype == SnrModelType.spncnn:
    g_model_class_name = SPNCNNModel



