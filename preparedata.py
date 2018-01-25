import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import math
import vis_utils


class FLAGS(object):
    image_height = 100
    image_width = 100
    image_channel = 1
    
    CORRECT_ORIENTATION = True
    
        
    
class PrepareData():
    def __init__(self):
        
        return
    def sparse_tuple_from_label(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []
    
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)
    
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    
        return indices, values, shape
    def preprocess_samples(self, samples):
        batch_inputs = []
        batch_labels = []

        for sample in samples:
            im,label = sample[:FLAGS.image_height * FLAGS.image_width], sample[FLAGS.image_height * FLAGS.image_width:]
            label = label.tolist()
            im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
            batch_inputs.append(im)
            batch_labels.append(label)
            
        
        res = [batch_inputs]
        if self.prepare_get_sparselabel:
            res.append(self.sparse_tuple_from_label(batch_labels))
        if self.parepare_get_denselabel:
            res.append(batch_labels)
       
        return res
    
    def __generator(self, samples, batch_size,is_training=True):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            if is_training:
                #during traning, shuffle the whole samples at the beginningof the epoch
                samples = shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                if is_training and (offset+batch_size > num_samples -1 ):
                    # this is to make sure all the batch are of same sizes during training
                    continue
                batch_samples = samples[offset:offset+batch_size]
                yield self.preprocess_samples(batch_samples)

    
    def get_samples(self, split_name):
        mnist_sequence = "./data/mnist_sequence3_sample_8distortions_9x9.npz"
        data = np.load(mnist_sequence)

        x_train, y_train = data['X_train'].reshape((-1, FLAGS.image_height * FLAGS.image_width)), data['y_train']
        x_valid, y_valid = data['X_valid'].reshape((-1, FLAGS.image_height * FLAGS.image_width)), data['y_valid']
        x_test, y_test = data['X_test'].reshape((-1, FLAGS.image_height * FLAGS.image_width)), data['y_test']
        
        if split_name == "train":
            res = np.concatenate([x_train, y_train], axis=1)
        elif split_name == "sample_test":
            res = np.concatenate([x_train[:100], y_train[:100]], axis=1)
        elif split_name == "eval":
            res = np.concatenate([x_valid, y_valid], axis=1)
        else:
            res = np.concatenate([x_test, y_test], axis=1)

        return res
        
    def input_batch_generator(self, split_name, is_training=False, batch_size=32, get_filenames = False, get_sparselabel = True, get_denselabel = True):
        samples = self.get_samples(split_name) 
        self.prepare_get_filenames = get_filenames
        self.prepare_get_sparselabel = get_sparselabel
        self.parepare_get_denselabel = get_denselabel
        gen = self.__generator(samples, batch_size, is_training=is_training)
       
        return gen, len(samples)
   
    def run(self):

        batch_size = 32
        split_name = 'sample_test'
#         split_name = 'train'
#         split_name = 'eval'
        generator, dataset_size = self.input_batch_generator(split_name, is_training=True, batch_size=batch_size, get_filenames=True,get_sparselabel = True)
        num_batches_per_epoch = int(math.ceil(dataset_size / float(batch_size)))
        for _ in range(num_batches_per_epoch):
            batch_inputs, batch_labels_sparse, batch_labels = next(generator)
            batch_inputs = np.array(batch_inputs)
            print(batch_labels)
            print("batch_size={}".format(len(batch_labels)))
            vis = True
            if vis:
                grid = vis_utils.visualize_grid(batch_inputs[:4])
                grid = np.squeeze(grid)
                plt.imshow(grid, cmap='gray')
                plt.show()
                break
                
        return
    
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()