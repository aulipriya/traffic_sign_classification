# Traffic Sign Classification 
This project's aim was to solve a simple classification problem using tensorflow.

### Dataset
This project uses [this](https://btsd.ethz.ch/shareddata/) dataset of traffic sign images. There are total 62 classes of images. Images were of different sizes, they were preprocessed to size 28,28 and converted to grayscale.


### Model 

The model is a very simple one with 3 convolution layer, first two followed by maxpool layer, then the output of convolution layers are flattened and fed to a two dense layer, last layer being the softmax output.

### Output

The model achieves 98.75% training accuracy on 10 epochs and 93.53% test accuracy 