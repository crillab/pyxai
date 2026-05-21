#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import pandas
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)    
    
    #
# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = './'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


#Take all 8 and 3 images to do a binary classifier 


x_train_8 = []
for i in range(len(x_train)):
    if y_train[i] == 8:
        x_train_8.append(x_train[i])

x_train_3 = []
for i in range(len(x_train)):
    if y_train[i] == 3:
        x_train_3.append(x_train[i])

 
show_images(x_train_8[0:5]+x_train_3[0:5], ["10"]*10)

rawdata = []
for i in range(len(x_train_3)):
    rawdata.append([v for l in x_train_3[i] for v in l]+[3])
for i in range(len(x_train_8)):
    rawdata.append([v for l in x_train_8[i] for v in l]+[8])

print("len:", len(rawdata[0]))

dataframe = pandas.DataFrame(rawdata)
dataframe.columns = ["pixel_"+str(i) for i in range(len(rawdata[0])-1)]+["label"]
#dataframe.to_csv("misnt_3_8.csv", index=False)
#print(dataframe)

# Create the dataset
from pyxai import Learning, Explainer, Tools

import datetime
import pandas

preprocessor = Learning.TabularPreprocessor(dataframe, target_feature="label", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS) 
print("data:", preprocessor.data)

preprocessor.all_numerical_features()

preprocessor.process() 
preprocessor.export("misnt_3_8")
