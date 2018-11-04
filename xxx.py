from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

img_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

image_path = './images/1-159.jpg'
image = np.reshape(np.asarray(Image.open(image_path), dtype='float32'), [-1, 128, 128, 3])
# img = image.load_img(img_path)
gen = img_generator.flow(image, batch_size=3)

# for x in next(gen):
#     print(x.shape)

while True:
    y = next(gen)
    print(y.shape)
    print(y)
    print(type(y))
    y = y.astype(np.uint8)
    
    im = Image.fromarray(np.reshape(y, [128, 128, 3]))
    im.save("your_file.jpeg")