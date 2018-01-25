import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from train_model_base import TrainModelBase
from preparedata import PrepareData
from nets.spncnn_model import SPNCNNModel
from snrglobaldef import g_modellogdir,g_snrmodeltype,SnrModelType
import math
class TrainModel(TrainModelBase):
    
    def config_training(self):
        self.batch_size = 32
        data_prep = PrepareData()
        self.train_feeder, self.num_train_samples = data_prep.input_batch_generator('sample_test', is_training=True, batch_size = self.batch_size, get_denselabel = False)
        print('get training image: ', self.num_train_samples)
        
        if g_snrmodeltype == SnrModelType.spncnn:
            self.train_dir = g_modellogdir
            self.max_number_of_epochs = 25
            self.save_cpt_epochs = 5 # save every self.save_cpt_epochs epochs
            self.save_summaries_steps = 500
            self.checkpoint_path = None
            self.checkpoint_exclude_scopes = None
            self.trainable_scopes = None
            
            self.learning_rate = 1e-3
            self.learning_rate_decay_type = 'fixed'
            self.optimizer = 'adam'
            self.opt_epsilon = 1e-8

        return
    def setup_model(self):
        if g_snrmodeltype == SnrModelType.spncnn:
            model = SPNCNNModel('train')
        
        model.build_graph()
        
        return model
    def get_next_feed(self, model):
        batch_inputs, batch_labels= next(self.train_feeder)
        feed = {model.inputs: batch_inputs,
                model.labels: batch_labels}
        return feed
    


if __name__ == "__main__":   
    obj= TrainModel()
    obj.run()