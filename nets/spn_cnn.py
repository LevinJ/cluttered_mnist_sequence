# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
from nets.transformer import spatial_transformer_network

def spn_cnn(inputs,
                     num_classes=10,
                     dropout_keep_prob=0.999,
                     is_training=True,
                     prediction_fn=tf.contrib.layers.softmax,
                     reuse=None,
                     scope='spncnn'):
    
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                                         len(input_shape))
    end_points = {}    
    with tf.variable_scope(scope, [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                                                is_training=is_training):
            ##############################
            #   LOCALIZATION NETWORK
            ##############################
            net = inputs
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net, 20, [3, 3], scope='l_conv0_loc')
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net, 20, [3, 3], scope='l_conv1_loc')
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net, 20, [3, 3], scope='l_conv2_loc')
            
            net = slim.flatten(net)
            net = slim.fully_connected(net, 200, scope='l_dense_loc')
            
            # INIT TRANSFORM TO IDENTITY
            b = np.zeros((2, 3), dtype='float32')
            b[0, 0] = 1
            b[1, 1] = 1
            
            theta = slim.fully_connected(net, 6, scope='A_net',weights_initializer = tf.zeros_initializer(), 
                                       biases_initializer=tf.constant_initializer(b.flatten()),activation_fn=None)
            input_fmap = inputs
            
            H = input_shape[1]
            W = input_shape[2]
#             out_dims = [tf.cast(H/2, tf.int32),tf.cast( W/2, tf.int32)]
            out_dims = [int(H/2), int(W/2)]
            net = spatial_transformer_network(input_fmap, theta, out_dims=out_dims)
            
            
            #classification network
            net = slim.conv2d(net, 96, [3, 3])
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net, 96, [3, 3])
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net, 96, [3, 3])
            net = slim.conv2d(net, 96, [3, 3])
            
            net = slim.flatten(net)
            net = slim.fully_connected(net, 400)
            
            logits = slim.fully_connected(net, num_classes,activation_fn=None)
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
            
            
    return logits, end_points


def ffn_spn_arg_scope(is_training=True,
                                                     weight_decay=0.00004,
                                                     stddev=0.09,
                                                     regularize_depthwise=False,
                                                     batch_norm_decay = 0.9997):
    """Defines the default ffn_spn arg scope.

    Args:
        is_training: Whether or not we're training the model.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: The standard deviation of the trunctated normal weight initializer.
        regularize_depthwise: Whether or not apply regularization on depthwise.

    Returns:
        An `arg_scope` to use for the ffn_spn model.
    """
    batch_norm_params = {
            'is_training': is_training,
            'center': True,
            'scale': True,
            'decay': batch_norm_decay,
            'epsilon': 0.001,
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.contrib.layers.xavier_initializer()
    regularizer = None
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected],
                                            weights_initializer=weights_init,
                                            activation_fn=tf.nn.relu, normalizer_fn=None):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                                        weights_regularizer=depthwise_regularizer) as sc:
                    return sc
                
                
                