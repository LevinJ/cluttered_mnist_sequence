import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import tensorflow as tf
import nets.spn_cnn as spn_cnn
from tensorflow.contrib import slim
import snr_mtrics as  snr_mtrics



class SPNCNNModel(object):
    def __init__(self, mode):
        self.mode = mode
        self.NUM_DIGITS = 3
        self.NUM_CLASSES = 10
        #variables to set before training
        self.inputs = tf.placeholder(tf.float32, [None, None, None, 1])
        self.labels = tf.placeholder(tf.int32, [None, self.NUM_DIGITS])
        return
    def build_graph(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        if self.mode == "train":
            self.build_train_graph()
        else:
            self.build_eval_graph() 
        return
    def build_eval_graph(self):
        self.add_inference_node(is_training = False)
        self.add_evalmetrics_node()
        return
    def add_evalmetrics_node(self):
        if self.labels is None:
            return
        predictions = self.prediction_result
        labels = self.labels
    
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/characteracc': snr_mtrics.streaming_character_accuracy(predictions, labels),
            'eval/wordacc': snr_mtrics.streaming_word_accuracy(predictions, labels)
        })
        for metric, value in names_to_values.items():
            value = tf.Print(value, [value], metric)
            tf.summary.scalar(metric, value)
        self.names_to_updates = list(names_to_updates.values())
        return 
    def build_train_graph(self):
        #before building the graph, we need to specify the input and labels, and variables_to_train for the models
        self.add_inference_node(is_training = True)
        self.add_loss_node()
        predictions = self.prediction_result
        labels = self.labels
       
        
        character_acc = snr_mtrics.character_accuracy(predictions, labels)
        word_acc = snr_mtrics.word_accuracy(predictions, labels)
        character_acc = tf.Print(character_acc, [character_acc], "character_accuracy")
        tf.summary.scalar('train_character_accuracy', character_acc)
        word_acc = tf.Print(word_acc, [word_acc], "word_acc")
        tf.summary.scalar('train_word_acc', word_acc)
            
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('image',self.inputs)
        return
    def prediction_funtion(self, logits, scope='Predictions'):
       
        logits= tf.reshape(logits, [-1, self.NUM_DIGITS, self.NUM_CLASSES])
        with tf.variable_scope(scope):
            predictions = tf.contrib.layers.softmax(logits)
            #add final final predition result
            self.prediction_result = tf.cast(tf.argmax(predictions, axis=-1),tf.int32)
            self.prediction_probability = predictions
        return predictions
    def add_inference_node(self, is_training=True):
        
        with slim.arg_scope(spn_cnn.ffn_spn_arg_scope()):
            self.output, self.end_points = spn_cnn.spn_cnn(self.inputs, num_classes= self.NUM_DIGITS * self.NUM_CLASSES, 
                                                                 is_training=is_training,  
                                                                 reuse=None,
                                                                 scope='spncnn')

            self.output = tf.reshape(self.output, [-1, self.NUM_DIGITS, self.NUM_CLASSES])
            
        return
    def add_loss_node(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels, logits = self.output)
        loss = tf.reduce_sum(loss, axis = -1)
        self.loss = tf.reduce_mean(loss)
        
        return
    
    