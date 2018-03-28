## Description
This is a implementation of cnn+triplet-loss for face identification written by keras, which is the last step of my **FaceID system**. You can find another two repositories  as follows:
1. [Face-detection-with-mobilenet-ssd](https://github.com/bruceyang2012/Face-detection-with-mobilenet-ssd)
2. [Face-Alignment-with-simple-cnn](https://github.com/bruceyang2012/Face-Alignment-with-simple-cnn)
3. [Face-identification-with-cnn-triplet-loss](https://github.com/bruceyang2012/Face-identification-with-cnn-triplet-loss)

## prepare data
1. Download [caltech faces](http://www.vision.caltech.edu/Image_Datasets/faces/faces.tar) from Official Website , and put it into face_data folder in [organize_data.py](https://github.com/bruceyang2012/Face-identification-with-cnn-triplet-loss/blob/master/utils/organize_data.py).
2. Run organize_data.py(https://github.com/bruceyang2012/Face-identification-with-cnn-triplet-loss/blob/master/utils/organize_data.py) to generate train, test, dev data.
3. Run [load_data.py](https://github.com/bruceyang2012/Face-identification-with-cnn-triplet-loss/blob/master/utils/load_data.py) to generate train_x.npy, train_y.npy and so on.

## train
Follow [face_train.ipynb](https://github.com/bruceyang2012/Face-identification-with-cnn-triplet-loss/blob/master/face_train.ipynb) step by step. You can change the parameters for better performance.

## References
[meownoid/face-identification-tpe](https://github.com/meownoid/face-identification-tpe)
