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
  <img src="" />
</p>
