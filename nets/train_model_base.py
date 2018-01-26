import os
import time
import tensorflow as tf

import math
import tensorflow.contrib.slim as slim

class FLAGS(object):
    decay_steps = 10000
    decay_rate = 0.98
    momentum = 0.9
    beta1 = 0.9
    beta2 = 0.999



class TrainModelBase():
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
       
        self.adadelta_rho = 0.95
        self.opt_epsilon= 1.0
        self.adagrad_initial_accumulator_value= 0.1
        self.adam_beta1= 0.9
        self.adam_beta2= 0.999
        self.ftrl_learning_rate_power = -0.5
        self.ftrl_initial_accumulator_value = 0.1
        self.ftrl_l1= 0.0
        self.ftrl_l2 = 0.0
        self.momentum= 0.9
        self.rmsprop_decay = 0.9
        self.rmsprop_momentum = 0.9
        self.label_smoothing = 0
        self.num_epochs_per_decay = 2.0
        self.end_learning_rate =  0.0001    
        
        self.save_cpt_epochs = 5 # save every self.save_cpt_epochs epochs
        self.save_summaries_steps = 500
        self.train_dir = './logs'
        self.batch_size= 32
        #optimiser
        self.optimizer = 'rmsprop'
        self.learning_rate = 1e-3
        self.learning_rate_decay_type = 'fixed' 
        self.max_number_of_epochs = None
        self.checkpoint_path = None
        self.checkpoint_exclude_scopes = None
        self.ignore_missing_vars = False
        return
    def config_training(self):
        pass
    def setup_model(self):
        pass
    def get_next_feed(self, model):
        pass
    def __configure_learning_rate(self, num_samples_per_epoch, global_step):
        """Configures the learning rate.
    
        Args:
            num_samples_per_epoch: The number of samples in each epoch of training.
            global_step: The global_step tensor.
    
        Returns:
            A `Tensor` representing the learning rate.
    
        Raises:
            ValueError: if
        """
        decay_steps = int(num_samples_per_epoch / self.batch_size *
                                            self.num_epochs_per_decay)
       
    
        if self.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(self.learning_rate,
                                                                                global_step,
                                                                                decay_steps,
                                                                                self.learning_rate_decay_factor,
                                                                                staircase=True,
                                                                                name='exponential_decay_learning_rate')
        elif self.learning_rate_decay_type == 'fixed':
            return tf.constant(self.learning_rate, name='fixed_learning_rate')
        elif self.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(self.learning_rate,
                                                                             global_step,
                                                                             decay_steps,
                                                                             self.end_learning_rate,
                                                                             power=1.0,
                                                                             cycle=False,
                                                                             name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized',
                                         self.learning_rate_decay_type)
        return
    def __configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.
    
        Args:
            learning_rate: A scalar or `Tensor` learning rate.
    
        Returns:
            An instance of an optimizer.
    
        Raises:
            ValueError: if FLAGS.optimizer is not recognized.
        """
        if self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate,
                    rho=self.adadelta_rho,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                    learning_rate,
                    initial_accumulator_value=self.adagrad_initial_accumulator_value)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    beta1=self.adam_beta1,
                    beta2=self.adam_beta2,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                    learning_rate,
                    learning_rate_power=self.ftrl_learning_rate_power,
                    initial_accumulator_value=self.ftrl_initial_accumulator_value,
                    l1_regularization_strength=self.ftrl_l1,
                    l2_regularization_strength=self.ftrl_l2)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=self.momentum,
                    name='Momentum')
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=self.rmsprop_decay,
                    momentum=self.rmsprop_momentum,
                    epsilon=self.opt_epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self.optimizer)
        return optimizer
#     def build_train_op(self, net):
# 
# #         net.lrn_rate = tf.train.exponential_decay(self.learning_rate,
# #                                                    net.global_step,
# #                                                    FLAGS.decay_steps,
# #                                                    FLAGS.decay_rate,
# #                                                    staircase=True)
# #         tf.summary.scalar("lerning_rate", net.lrn_rate)
# 
#         # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=net.lrn_rate,
#         #                                            momentum=FLAGS.momentum).minimize(self.cost,
#         #                                                                              global_step=self.global_step)
#         # self.optimizer = tf.train.MomentumOptimizer(learning_rate=net.lrn_rate,
#         #                                             momentum=FLAGS.momentum,
#         #                                             use_nesterov=True).minimize(self.cost,
#         #                                                                         global_step=self.global_step)
# 
#         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
#                                                 beta1=FLAGS.beta1,
#                                                 beta2=FLAGS.beta2).minimize(net.loss,
#                                                                             global_step=net.global_step)
#         train_ops = [optimizer] + net._extra_train_ops
#         net.train_op = tf.group(*train_ops)
#         return
    def __get_variables_to_train(self):
        """Returns a list of variables to train.
    
        Returns:
            A list of variables to train by the optimizer.
        """
        if self.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self.trainable_scopes.split(',')]
    
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train
    def warm_start_training(self, sess):
        """Returns a function run by the chief worker to warm-start the training.
    
        Note that the init_fn is only run when initializing the model during the very
        first global step.
    
        Returns:
            An init function run by the supervisor.
        """  
        ckpt = tf.train.latest_checkpoint(self.train_dir)
        if ckpt:
            # the global_step will restore sa well
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            saver.restore(sess, ckpt)
            print('restore from the checkpoint{0}'.format(ckpt))
            return
        
        if self.checkpoint_path is None:
            return None
    
        exclusions = []
        if self.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                                        for scope in self.checkpoint_exclude_scopes.split(',')]
        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        all_variables = slim.get_model_variables()
        if tf.gfile.IsDirectory(self.checkpoint_path):
            global_step = slim.get_or_create_global_step()
            all_variables.append(global_step)
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path
            
        for var in all_variables:
            excluded = False
             
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
    
#         tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    
        slim.assign_from_checkpoint_fn(
                checkpoint_path,
                variables_to_restore)(sess)
        return
    def run(self):
        self.config_training()
        model = self.setup_model()
        
        num_steps_per_epoch = int(math.ceil(self.num_train_samples / float(self.batch_size)))
        max_number_of_steps = self.max_number_of_epochs * num_steps_per_epoch
        print("max steps = {}, max epochs = {}, num_steps_per_epoch = {}, batch_size={}".format(max_number_of_steps, 
                                                                                                  self.max_number_of_epochs, num_steps_per_epoch, self.batch_size))
        
        
        # set up optimiser for the model
        learning_rate = self.__configure_learning_rate(self.num_train_samples, model.global_step)
        optimizer = self.__configure_optimizer(learning_rate)
        variables_to_train = self.__get_variables_to_train()
        model.train_op  = slim.learning.create_train_op(model.loss, optimizer, variables_to_train=variables_to_train, global_step= model.global_step)
        if hasattr(model, 'extra_train_ops'):
            temp_train_ops = [model.train_op] + model.extra_train_ops
            model.train_op= tf.group(*temp_train_ops)
#         
        #set up summary
        model.merged_summay = tf.summary.merge_all()
        
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.warm_start_training(sess)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            train_writer = tf.summary.FileWriter(self.train_dir + '/trains', sess.graph)
            
            last_global_step = sess.run(model.global_step)  
            print("last_global_step={}".format(last_global_step)) 
           
           
            for _ in range(last_global_step, max_number_of_steps):
               
                batch_time = time.time()
                
                feed = self.get_next_feed(model)
                loss, step, _ = sess.run([model.loss, model.global_step, model.train_op], feed)
                
                if step % 100 == 0:
                    print('{}/{},{}/{}, loss={}, time={}'.format(step, max_number_of_steps, int(step/num_steps_per_epoch) + 1, self.max_number_of_epochs, loss, time.time() - batch_time))
    
                # monitor trainig process
                if step % self.save_summaries_steps == 0 or (step == max_number_of_steps):
                    
                    feed = self.get_next_feed(model)
                    summary_str = sess.run(model.merged_summay,feed)
                    train_writer.add_summary(summary_str, step)
                    
                # save the checkpoint once very few epoochs
                if (step % (self.save_cpt_epochs *num_steps_per_epoch)  == 0) or (step == max_number_of_steps):
                    if not os.path.isdir(self.train_dir):
                        os.mkdir(self.train_dir)
                    print('save the checkpoint of step {}'.format(step))
                    saver.save(sess, self.train_dir + '/model.ckpt', global_step=step)
            return


if __name__ == "__main__":   
    obj= TrainModelBase()
    obj.run()
                    


