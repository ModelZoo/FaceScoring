# Emotion Recognition

Emotion Recognition Implemented by [ModelZoo](https://github.com/ModelZoo/ModelZoo).

## Usage

Firstly, you need to clone this repository and download training data:

```
git clone https://github.com/ModelZoo/EmotionRecognition.git
cd EmotionRecognition
git lfs pull
```

Next, install the dependencies using pip:

```
pip3 install -r requirements.txt
```

Finally, just run training:

```
python3 train.py
```

If you want to continue training your model, you need to define `checkpoint_restore` flag in `train.py`:

```python
tf.flags.DEFINE_bool('checkpoint_restore', True, help='Model restore')
```

And you can define the specific model with `checkpoint_name` which you want to continue training with:

```python
tf.flags.DEFINE_string('checkpoint_name', 'model-178.ckpt', help='Model name')
```


## TensorBoard

After training, you can see the transition of loss in TensorBoard.

```
cd events
tensorboard --logdir=.
```

![](https://ws3.sinaimg.cn/large/006tNbRwgy1fw37u664tzj319d0mumym.jpg)

The best accuracy is 65.64% from step 178.

## Predict

Next, we can use our model to recognize the emotion.

Here are the test pictures we picked from the website:

![](https://ws4.sinaimg.cn/large/006tNbRwgy1fw3f6am6jpj310405cwf8.jpg)

Then put them to the folder named `tests` and define the
 model path and test folder in `infer.py`:

```python
tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt-178', help='Model name')
tf.flags.DEFINE_string('test_dir', 'tests/', help='Dir of test data')
```

Then just run inference using this cmd:

```
python3 infer.py
```

We can get the result of emotion recognition and probabilities of each emotion:

```
Image Path: test1.png
Predict Result: Happy
Emotion Distribution: {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 1.0, 'Sad': 0.0, 'Surprise': 0.0, 'Neutral': 0.0}
====================
Image Path: test2.png
Predict Result: Happy
Emotion Distribution: {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.998, 'Sad': 0.0, 'Surprise': 0.0, 'Neutral': 0.002}
====================
Image Path: test3.png
Predict Result: Surprise
Emotion Distribution: {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 1.0, 'Neutral': 0.0}
====================
Image Path: test4.png
Predict Result: Angry
Emotion Distribution: {'Angry': 1.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0, 'Neutral': 0.0}
====================
Image Path: test5.png
Predict Result: Fear
Emotion Distribution: {'Angry': 0.04, 'Disgust': 0.002, 'Fear': 0.544, 'Happy': 0.03, 'Sad': 0.036, 'Surprise': 0.31, 'Neutral': 0.039}
====================
Image Path: test6.png
Predict Result: Sad
Emotion Distribution: {'Angry': 0.005, 'Disgust': 0.0, 'Fear': 0.027, 'Happy': 0.002, 'Sad': 0.956, 'Surprise': 0.0, 'Neutral': 0.009}
```

Emmm, looks good!

## Pretrained Model

Looking for pretrained model?

just go to `checkpoints` folder, here is the model with best performance at step 178.
