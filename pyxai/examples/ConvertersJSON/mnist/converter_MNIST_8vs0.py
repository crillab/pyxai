import pandas as pd
from PIL import Image
from tqdm import tqdm 
import os
import idx2numpy

from pyxai import Learning

path = os.sep.join(__file__.split(os.sep)[:-1])+os.sep
if not os.path.isdir(path+'Images'):
    os.mkdir(path+'Images')

preprocessor = Learning.NonTabularPreprocessor('classification', 'image', 'classes')

target_label = 8
extra_labels = [0] # Keep empty if you want to keep all the data

def create_df_and_images(img_idx_path, label_idx_path, start_idx, img_folder):
    imgs_arr = idx2numpy.convert_from_file(img_idx_path)
    test_label_arr = idx2numpy.convert_from_file(label_idx_path)
    data_df = pd.DataFrame(columns=['file_name', 'label'])
    for i in tqdm(range(len(imgs_arr))):
        im_i = Image.fromarray(imgs_arr[i])
        file_name = "image_{i}.png".format(i=i+start_idx)
        im_i.save(img_folder+'/'+file_name)
        data_df.loc[i] = [file_name, test_label_arr[i]]
    return data_df

test_df = create_df_and_images(path+'t10k-images-idx3-ubyte', path+'t10k-labels-idx1-ubyte', 0, path+'Images')
train_df = create_df_and_images(path+'train-images-idx3-ubyte', path+'train-labels-idx1-ubyte',len(test_df), path+'Images')
kept_labels = extra_labels + [target_label]

instance_id = 0 # Same instance_id for the features and the labels
for _, row in train_df.iterrows():  
    if (len(kept_labels)<=1) or (row.label in kept_labels):
        preprocessor.add_instance_image(instance_id=instance_id, file_path='Images'+os.sep+row.file_name, instances_set=Learning.TRAIN)
        preprocessor.add_label_class(instance_id=instance_id, label=row.label)
        instance_id += 1

for _, row in test_df.iterrows():
    if (len(kept_labels)<=1) or (row.label in kept_labels):
        preprocessor.add_instance_image(instance_id=instance_id, file_path='Images'+os.sep+row.file_name, instances_set=Learning.TEST)
        preprocessor.add_label_class(instance_id=instance_id, label=row.label)
        instance_id += 1

preprocessor.to_json(path+'mnist_8vs0.json')
