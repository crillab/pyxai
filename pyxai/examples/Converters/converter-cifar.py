# dataset source: http://www.cs.toronto.edu/~kriz/cifar.html (python version)

import pickle
import os
import pandas
import numpy 

path="../../cifar-10-batches-py/"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dataset_to_dict():
    for root, dirs, files in os.walk(path):
        for name in files:
            if name == "data_batch_1":
                return unpickle(root + name)

dataset = dataset_to_dict()

keys_cats = [k for k in range(len(dataset[b'labels'])) if dataset[b'labels'][k] == 3]
keys_dogs = [k for k in range(len(dataset[b'labels'])) if dataset[b'labels'][k] == 5]

rawdata = []
for key in keys_cats:
    rawdata.append(numpy.append(dataset[b'data'][key], [dataset[b'labels'][key]]))
for key in keys_dogs:
    rawdata.append(numpy.append(dataset[b'data'][key], [dataset[b'labels'][key]]))

dataframe = pandas.DataFrame(rawdata)
dataframe.columns = [str(i) for i in range(len(dataset[b'data'][key]))]+["label"]
dataframe.to_csv("examples/datasets_not_converted/cifar_cat_dog.csv", index=False)
print(dataframe)