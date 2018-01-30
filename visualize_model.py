import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
import tensorflow as tf
from preparedata import PrepareData
import math
import argparse
from snrglobaldef import g_modellogdir, g_model_class_name
import matplotlib.pyplot as plt
import numpy as np
import cv2

class VisualizeModel(PrepareData):
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
        batch_size = 1
        model = g_model_class_name('eval')
        model.build_graph()
        model.merged_summay = tf.summary.merge_all()
        val_feeder, num_samples = self.input_batch_generator(self.split_name, is_training=False, batch_size = batch_size, get_sparselabel = False)
       
        num_batches_per_epoch = int(math.ceil(num_samples / float(batch_size)))
       
      
    
        
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
    
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            
            
            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            print('Evaluating checkpoint_path={}, split={}, num_samples={}'.format(checkpoint_file, self.split_name, num_samples))
           
            saver.restore(sess, checkpoint_file)
           
    

            
            cnt = 0
            for _ in range(num_batches_per_epoch):
                inputs, labels= next(val_feeder)
                feed = {model.inputs: inputs,
                            model.labels: labels}
                
                prediction, st, x_s, y_s = sess.run([model.prediction_result, model.end_points['st'], model.end_points['x_s'], model.end_points['y_s']], feed)
                
                if (~(prediction[0] == labels[0])).sum() == 0:
                    cnt += 1
                    
                    if cnt != 4:
                        continue
                    print(prediction)
                    row, col = 1,3
                    input_img = 1.0 - np.squeeze(inputs[0])
                    st_img = 1.0 - np.squeeze(st)
                    masked_img = self.get_roi_visualization(input_img, x_s, y_s)
                    plt.subplot(row, col,1),plt.imshow(input_img, 'gray')
                    plt.subplot(row, col,2),plt.imshow(masked_img)
                    plt.subplot(row, col,3),plt.imshow(st_img, 'gray')
                    plt.show()
                    break;
                
#                 print('{}/{}, {:.5f} seconds.'.format(i, num_batches_per_epoch, elapsed))
                    
                # print the decode result
                
           
            return
    def get_roi_visualization(self, input_img, x_s, y_s):
#         x_s = np.squeeze(x_s)
#         y_s = np.squeeze(y_s)
#         R_channel = input_img.copy()
#         G_channel = input_img.copy()
#         B_channel = np.zeros_like(input_img)
#         
#         # here we assume that the spatial transformed image is of size 50 x 50
#         x_s =  (x_s * 50 + 50).astype(np.int32)
#         y_s =  (y_s * 50 + 50).astype(np.int32)
#         
#         for i in range(50):
#             for j in range(50):
#                 x_index = x_s[i,j]
#                 y_index = y_s[i,j]
#                 B_channel[y_index, x_index] = 255
#         res = np.dstack((R_channel,G_channel, B_channel))
        
        
        
        x_s = np.squeeze(x_s)
        y_s = np.squeeze(y_s)
        gray = ((1.0 - input_img) * 255).astype(np.uint8)
        res_img = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        plt.imshow(res_img[...,::-1])
        
        # here we assume that the spatial transformed image is of size 50 x 50
        x_s =  (x_s * 50 + 50).astype(np.int32)
        y_s =  (y_s * 50 + 50).astype(np.int32)
        
        for i in range(50):
            for j in range(50):
                x_index = x_s[i,j]
                y_index = y_s[i,j]
                cv2.circle(res_img,(x_index,y_index), 1, (0,0,255), -1)
                plt.imshow(res_img[...,::-1])
       
        plt.imshow(res_img[...,::-1])
        res_img = res_img[...,::-1]
        return res_img
    def run(self):
        self.parse_param()
        self.eval_model()
        return

       




if __name__ == "__main__":   
    obj= VisualizeModel()
    obj.run()
