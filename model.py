from tensorflow.python.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.python.keras.metrics import categorical_accuracy
# from tensorflow.losses import mean_squared_error
from tensorflow.losses import mean_pairwise_squared_error

from model_zoo.model import BaseModel
import tensorflow as tf
import numpy as np


class VGGModel(BaseModel):
    def __init__(self, config):
        super(VGGModel, self).__init__(config)
        self.num_features = 64
        # layer1
        self.conv11 = tf.keras.layers.Conv2D(filters=self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.conv12 = tf.keras.layers.Conv2D(filters=self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop1 = tf.keras.layers.Dropout(rate=0.5)
        
        # layer2
        self.conv21 = tf.keras.layers.Conv2D(filters=2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.conv22 = tf.keras.layers.Conv2D(filters=2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop2 = tf.keras.layers.Dropout(rate=0.5)
        
        # layer3
        self.conv31 = tf.keras.layers.Conv2D(filters=2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.conv32 = tf.keras.layers.Conv2D(filters=2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.conv33 = tf.keras.layers.Conv2D(filters=2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu',
                                             padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop3 = tf.keras.layers.Dropout(rate=0.5)
        
        # layer4
        self.conv41 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.conv42 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.conv43 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop4 = tf.keras.layers.Dropout(rate=0.5)
        
        # layer5
        self.conv51 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.conv52 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.conv53 = tf.keras.layers.Conv2D(filters=2 * 2 * 2 * self.num_features, kernel_size=(3, 3),
                                             activation='relu',
                                             padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop5 = tf.keras.layers.Dropout(rate=0.5)
        
        # flatten
        self.flatten = tf.keras.layers.Flatten()
        
        # dense
        self.dense1 = tf.keras.layers.Dense(2 * 2 * 2 * self.num_features, activation='relu')
        self.drop5 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(2 * 2 * self.num_features, activation='relu')
        self.drop6 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(2 * self.num_features, activation='relu')
        self.drop7 = tf.keras.layers.Dropout(0.5)
        
        self.dense4 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=None, mask=None):
        # layer1
        x = self.conv11(inputs)
        x = self.conv12(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        # layer2
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        # layer3
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.drop3(x, training=training)
        # layer4
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        x = self.drop4(x, training=training)
        # layer5
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.bn5(x, training=training)
        x = self.pool5(x)
        x = self.drop5(x, training=training)
        
        # flatten
        x = self.flatten(x)
        # dense
        x = self.dense1(x)
        x = self.drop5(x, training=training)
        x = self.dense2(x)
        x = self.drop6(x, training=training)
        x = self.dense3(x)
        x = self.drop7(x, training=training)
        x = self.dense4(x)
        return x
    
    def optimizer(self):
        return tf.train.AdamOptimizer(self.config.get('learning_rate'))
    
    # def loss(self, y_true, y_pred):
    #     print('y_true', y_true.shape, 'y_pred', y_pred.shape)
    #     print('Y_true[0]', y_true[0], 'y_pred[0]', y_pred[0])
    #     y_true_argmax, y_pred_argmax = tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)
    #     print('y_true_argmax', y_true_argmax.shape, 'y_pred_argmax', y_pred_argmax.shape)
    #     print('y_true_argmax[0]', y_true_argmax[0], 'y_pred_argmax', y_pred_argmax[0])
    #     result = tf.reduce_mean(tf.square(y_pred_argmax - y_true_argmax))
    #     print('Result', result)
    #     return result
    
    def loss(self, y_true, y_pred):
        return mean_pairwise_squared_error(y_true, y_pred)
    
    def mse(self, y_true, y_pred):
        y_true_argmax, y_pred_argmax = tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)
        return mean_squared_error(y_true_argmax, y_pred_argmax)
    
    def accuracy(self, y_true, y_pred):
        y_true_argmax, y_pred_argmax = tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)
        return tf.reduce_mean(tf.cast(tf.equal(y_true_argmax, y_pred_argmax), tf.float32))
    
    def init(self):
        self.compile(optimizer=self.optimizer(),
                     loss=self.loss,
                     metrics=[self.mse, self.accuracy])
    
    def infer(self, test_data, batch_size=None):
        logits = self.predict(test_data)
        preds = np.argmax(logits, axis=-1)
        return logits, preds
