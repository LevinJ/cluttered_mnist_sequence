import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
import tensorflow as tf
from preparedata import PrepareData
import math
import argparse
from snrglobaldef import g_modellogdir, g_model_class_name

class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        return
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--split_name',  help='which split of dataset to use',  default="eval")
        parser.add_argument('-c', '--checkpoint_path',  help='which checkpoint to use',  default= '')
        args = parser.parse_args()
        
        if  args.checkpoint_path == 'finetune':
            self.checkpoint_path = "{}/{}".format(g_modellogdir, args.checkpoint_path)
        elif args.checkpoint_path == '':
            self.checkpoint_path = g_modellogdir
        else:
            self.checkpoint_path = args.checkpoint_path
            
        self.split_name = args.split_name
            
        return
    def eval_model(self):
        batch_size = 32
        model = g_model_class_name('eval')
        model.build_graph()
        model.merged_summay = tf.summary.merge_all()
        val_feeder, num_samples = self.input_batch_generator(self.split_name, is_training=False, batch_size = batch_size, get_sparselabel = False)
       
        num_batches_per_epoch = int(math.ceil(num_samples / float(batch_size)))
       
      
    
        
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
    
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            eval_writer = tf.summary.FileWriter("{}/evals/{}".format(g_modellogdir, self.split_name), sess.graph)
            
            
            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            print('Evaluating checkpoint_path={}, split={}, num_samples={}'.format(checkpoint_file, self.split_name, num_samples))
           
            saver.restore(sess, checkpoint_file)
           
    

            

            for _ in range(num_batches_per_epoch):
                inputs, labels= next(val_feeder)
                feed = {model.inputs: inputs,
                            model.labels: labels}
                start = time.time()
                _ = sess.run(model.names_to_updates, feed)
                elapsed = time.time()
                elapsed = elapsed - start
#                 print('{}/{}, {:.5f} seconds.'.format(i, num_batches_per_epoch, elapsed))
                    
                # print the decode result
                
            summary_str, step = sess.run([model.merged_summay, model.global_step])
            eval_writer.add_summary(summary_str, step)
            return
    def run(self):
        self.parse_param()
        self.eval_model()
        return

       




if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()
