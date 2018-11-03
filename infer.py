from model_zoo.inferer import BaseInferer
import tensorflow as tf
import cv2
from os import listdir
from os.path import join
import numpy as np

tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt-178', help='Model name')
tf.flags.DEFINE_string('test_dir', 'tests/', help='Dir of test data')


class Inferer(BaseInferer):
    
    def prepare_data(self):
        test_dir = self.flags.test_dir
        items = sorted(list(listdir(test_dir)))
        items_path = list(map(lambda x: join(test_dir, x), items))
        test_data = list(map(lambda x: self.process_image(x), items_path))
        test_data = np.asarray(test_data)
        self.items = items
        return test_data
    
    def process_image(self, image_file):
        image = cv2.imread(image_file, 0)
        image = cv2.resize(image, (48, 48))
        image_data = np.reshape(np.asarray(image), (48, 48, 1)).astype(np.float32)
        image_data /= 255.0
        return image_data


if __name__ == '__main__':
    emotion_cat = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    inferer = Inferer()
    logits, preds = inferer.run()
    for item, logit, pred in zip(inferer.items, logits, preds):
        result = emotion_cat[pred]
        print('=' * 20)
        print('Image Path:', item)
        print('Predict Result:', emotion_cat[pred])
        print('Emotion Distribution:', {v: round(logit[k], 3) for k, v in emotion_cat.items()})
