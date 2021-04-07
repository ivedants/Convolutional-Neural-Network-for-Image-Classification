# Convolutional Neural Network for Image Classification

This is an implementation of a Convolutional Neural Network for Image Classification on MNIST Dataset on Keras. 

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*XdCMCaHPt-pqtEibUfAnNw.png" />
</p>

## About the Dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits from 0 to 9 that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets.

Fortunately, this data is easy to access with Keras. The data set has:
- 60,000 training images
- 10,000 test images

## Diving Deep

A single digit image on the MNIST Dataset can be represented as an array. Specifically, 28 x 28 pixels. 

<p align="center">
  <img src="https://programmersought.com/images/543/027a10117fba65874743cbfc34f8c61f.png" />
</p>

The values represent the greyscale image. We can think of the entire group of the 60,000 images as a 4-dimensional array. 60,000 images of 1 channel of 28 x 28 pixels. 

For better understanding, this array can be represented as:
(Samples, x, y, channels) = (60000, 28, 28, 1)
#### NOTE: For color images, the last dimension value would be 3 (as they are represented in RGB values)

## Labels

For labels, we use One-Hot Encoding which allows the representation of categorical data to be more expressive. This means that instead of having labels such as "one", "two", etc., we will have a single array for each image. Many machine learning algorithms cannot work with categorical data directly. The categories must be converted into numbers. This is required for both input and output variables that are categorical. This type of categorical variable binary representation is called one-hot, because each row has one feature with a value of 1, and the other features with value 0.

So, what this means is that if the original labels of the images are given as a list of numbers, we convert them to one-hot encoding by simply using the function to_categorical from keras.utils.np_utils. 

The label is then represented based off the index position in the label array i.e. the corresponding label will be a 1 at the index location and 0 everywhere else. For example, a drawn digit of 4 would have this label array: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0].

As a result, the labels for the training data end up being a large 2-D array of the dimension (60000, 10). 

## Diving Deep into the Jupyter Notebook Code

After the imports, we take a look at the actual shape of the training data. So we get (60000, 28, 28). Right now, we don't have a color channel so we reshape this data to have a color channel. Then the next thing we do is just grab the very first sample by indexing at zero. 

<p align="center">
  <img src="https://github.com/ivedants/Convolutional-Neural-Network-for-Image-Classification/blob/main/Single%20sample.jpg" />
</p>

As we see, most of the values in the array are 0, which is actually representing the white color pixels of the image. 

### Data Pre-Processing

We first start with making the labels understandable for the CNN. Notice how the training data just returns 60,000 values indicating the actual number. If we feed the values this way for training our CNN, the network would probably get confused and think of the data as some sort of regression problem i.e. as if the values, say 5, 4, 0 are values on some sort of continuous scale instead of actual distinct categories. Hence, we convert them to One-Hot Encoding, as explained above. 

Doing this would convert all the values of the training and test data in the array into binary values i.e. categorical data which would be much more simpler for a Neural Network to understand.

In order to normalize the X data, we simply divide the values of the arrays by the max value in order to normalize this to be within 0 and 1. 

In order to make this into a generalized network that can work on just any sort of image data, we reshape the data to also include color channels (we consider one color channel in this specific case).

### Training the Model

After pre-processing the data, we create the sequential object of the model to start off with the convolutional layer. We add the convolutional layer with the kernel size of 4 x 4 and 32 filters, which are considered good standard values for this image dataset we are working with. Then we put in the input shape values and the activation function as rectified linear unit which works quite well for CNNs. 

After this, we add the Pooling Layer, followed by flattening the images from 28 x 28 to 764 pixels before the final layer. Then we add a dense layer by choosing the number of neurons in this hidden layer finally followed by an ultimate classifier layer, which would classify the images into 10 possible classes. For this, the activation function is softmax function so the output directly gives us a categorical class. 

### Evaluating the Model

The evaluation looks pretty good. The precision, recall, and F-1 scores are about 99 percent and that proves that our convolutional neural network can easily classify hand-written digits of the MNIST dataset with minimal error. 

