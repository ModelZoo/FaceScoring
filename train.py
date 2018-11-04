import tensorflow as tf
from model_zoo.trainer import BaseTrainer
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

tf.flags.DEFINE_string('data_dir', './images', help='Data dir')
tf.flags.DEFINE_float('learning_rate', 0.001, help='Learning rate')
tf.flags.DEFINE_integer('epochs', 1000, help='Max epochs')
tf.flags.DEFINE_integer('early_stop_patience', 500, help='Early stop patience')
tf.flags.DEFINE_bool('checkpoint_restore', False, help='Model restore')
tf.flags.DEFINE_string('model_class', 'VGGModel', help='Model class name')
tf.flags.DEFINE_integer('batch_size', 20, help='Batch size')
tf.flags.DEFINE_integer('enhance_images_number', 100, help='Enhance images number')


class Trainer(BaseTrainer):
    image_generator = image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    def enhance_image(self, image):
        """
        generate enhanced image
        :param image:
        :return:
        """
        image = np.expand_dims(image, axis=0)
        gen = self.image_generator.flow(image)
        for i in range(self.flags.enhance_images_number):
            enhanced_image = next(gen)
            enhanced_image = np.reshape(enhanced_image, enhanced_image.shape[1:])
            yield enhanced_image
    
    def prepare_data(self):
        """
        prepare data
        :return:
        """
        # read data
        x_data, y_data = [], []
        path_data = self.flags.data_dir
        for index, image in enumerate(listdir(path_data)):
            if index % 20 == 0:
                print('Processing', index + 1, 'Images')
            image_path = join(path_data, image)
            label = int(image.split('-')[0]) - 1
            image_data = Image.open(image_path)
            image_array = np.reshape(np.asarray(image_data, dtype='float32'), [128, 128, 3])
            x_data.append(image_array)
            y_data.append(label)
            # get enhanced images
            for image in self.enhance_image(image_array):
                x_data.append(image)
                y_data.append(label)
        
        # to numpy
        x_data, y_data = np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)
        print('X Data', x_data.shape, 'Y Data', y_data.shape)
        # build label to one hot
        y_data = tf.keras.utils.to_categorical(np.asarray(y_data).astype('float32'), 10)
        print('Len X', len(x_data), 'Len Y', len(y_data))
        # split data
        x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.15)
        print('Len X Train', len(x_train), 'Len X Eval', len(x_eval))
        print('Len Y Train', len(y_train), 'Len Y Eval', len(y_eval))
        return (x_train, y_train), (x_eval, y_eval)


if __name__ == '__main__':
    Trainer().run()
