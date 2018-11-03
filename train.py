import tensorflow as tf
from model_zoo.trainer import BaseTrainer
import numpy as np
import pandas as pd
from os import listdir
from os.path import join
from PIL import Image
from sklearn.model_selection import train_test_split

tf.flags.DEFINE_string('data_dir', './images', help='Data dir')
tf.flags.DEFINE_float('learning_rate', 0.001, help='Learning Rate')
tf.flags.DEFINE_integer('epochs', 1000, help='Max Epochs')
tf.flags.DEFINE_integer('early_stop_patience', 500, help='Early Stop Patience')
tf.flags.DEFINE_bool('checkpoint_restore', False, help='Model restore')
tf.flags.DEFINE_string('model_class', 'VGGModel', help='Model class name')
tf.flags.DEFINE_integer('batch_size', 20, help='Batch size')


class Trainer(BaseTrainer):
    
    def prepare_data(self):
        # read data
        x_data, y_data = [], []
        path_data = self.flags.data_dir
        for image in listdir(path_data):
            image_path = join(path_data, image)
            label = image.split('-')[0]
            image_data = Image.open(image_path)
            image_array = np.reshape(np.asarray(image_data, dtype='float32'), [128, 128, 3])
            x_data.append(image_array)
            y_data.append(label)
        # to numpy
        x_data, y_data = np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)
        print('X Data', x_data.shape, 'Y Data', y_data.shape)
        # build label to one hot
        y_data = tf.keras.utils.to_categorical(np.asarray(y_data).astype('float32'), 10)
        print('Len X', len(x_data), 'Len Y', len(y_data))
        # split data
        x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.20, random_state=42)
        return (x_train, y_train), (x_eval, y_eval)


if __name__ == '__main__':
    Trainer().run()
