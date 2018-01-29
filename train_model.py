import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from nets.train_model_base import TrainModelBase
from preparedata import PrepareData
from nets.spncnn_model import SPNCNNModel
from snrglobaldef import g_modellogdir,g_model_class_name
import math
class TrainModel(TrainModelBase):
    
    def config_training(self):
        self.batch_size = 100
        data_prep = PrepareData()
        self.train_feeder, self.num_train_samples = data_prep.input_batch_generator('train', is_training=True, batch_size = self.batch_size, get_sparselabel = False)
        print('get training image: ', self.num_train_samples)
        
        
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
        
        model = SPNCNNModel('train')
        model.build_graph()
        
        return model
    def get_next_feed(self, model):
        batch_inputs, batch_labels= next(self.train_feeder)
        feed = {model.inputs: batch_inputs,
                model.labels: batch_labels}
        return feed


class Train_SPNCNNModel(TrainModel):
    def config_training(self):
        TrainModel.config_training(self)
        self.max_number_of_epochs = 50
        return
    def setup_model(self):
        model = SPNCNNModel('train')
        if self.max_number_of_epochs <= 1:
            model.batch_norm_decay = 0.9
            print("adust batch norm decay {}".format(model.batch_norm_decay ))
        model.keep_prob = 0.75
        if self.max_number_of_epochs > 30:
            self.learning_rate = 1e-4
        model.build_graph()
        return model
    
    


if __name__ == "__main__":   
    train_class_name = "Train_{}".format(g_model_class_name.__name__)
    train_class_name = eval(train_class_name)
    obj= train_class_name()
    obj.run()