# HHAR
Comparision of Heterogeneous Human Activity Recognition using Feedforward Neural Network and Recurrent Neural Network

**About the project :**

Identification of physical activities performed by human subjects is referred to as Heterogeneous Human Activity Recognition (HHAR). In this project I propose to use recurrent neural network to predict human activities. The model uses the existing [UCI HHAR](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition) dataset which contains the accelerometer and gyroscope sensor readings in the x, y, and z directions carried out in smart watches and smartphones of 9 users. The project also compares the performance of different fully connected neural
network and Recursive neural network to find out the best fit architecture to obtain the highest accuracy. This aim of this project is to implement such a neural network on an accelerometer dataset and analyze results. The same model can be used for predicting gyroscope activity.

**How to run the project :**

_Note: Since the neural network training involves memory and processing intense tasks, i suggest you to use cloud notebooks like Google Colab._

First, download the dataset either by running the _downloadDataset.sh_ or if you are not using linux, download the dataset from [_here_](http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip) and extract the contents to the "dataset" folder.

**Requirements :**
1. Pytorch
2. Torchvision
3. Numpy
4. Pandas
5. Matplotlib

That's pretty much it. You are now ready to run the FNNvsRNN.ipynb notebook and have a look at the project.
