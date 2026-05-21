
from pyxai.sources.type import _check_parameter, ProblemType, LabelsType, InstancesType
import json

class NonTabularPreprocessor:
    r"""The ```NonTabularPreprocessor``` class is used to preprocess non tabular datasets.
    """

    def __init__(self, problem_type, instances_type, labels_type):
        r"""To create a JSON file representing a dataset. 

            Parameters
            ----------
            problem_type: ```str``` | ```ProblemType```
                The type of problem (classification, regression, ...)
                Possible values are defined in the ```ProblemType``` enum.
                    
            instances_type: ```str``` | ```InstancesType``` (optional, default=```None```)
                The type of instances (image, tabular, text, temporal, ...)
                Possible values are defined in the ```InstancesType``` enum.
                    
            labels_type: ```str``` | ```LabelsType``` (optional, default=```None```)
                The type of labels (class, text, mask, contours, ...)
                Possible values are defined in the ```LabelsType``` enum.

            Returns
            -------
            ```NonTabularPreprocessor```:
               A ```NonTabularPreprocessor``` object

            Examples
            --------
            >>> import pandas as pd
            >>> from PIL import Image
            >>> from tqdm import tqdm 
            >>> import os
            >>> import idx2numpy

            >>> from pyxai import Learning

            >>> path = os.sep.join(__file__.split(os.sep)[:-1])+os.sep
            >>> if not os.path.isdir(path+'Images'):
            >>>     os.mkdir(path+'Images')

            >>> preprocessor = Learning.NonTabularPreprocessor('classification', 'image', 'classes')

            >>> target_label = 8
            >>> extra_labels = [0] # Keep empty if you want to keep all the data

            >>> def create_df_and_images(img_idx_path, label_idx_path, start_idx, img_folder):
            >>>     imgs_arr = idx2numpy.convert_from_file(img_idx_path)
            >>>     test_label_arr = idx2numpy.convert_from_file(label_idx_path)
            >>>     data_df = pd.DataFrame(columns=['file_name', 'label'])
            >>>     for i in tqdm(range(len(imgs_arr))):
            >>>         im_i = Image.fromarray(imgs_arr[i])
            >>>         file_name = "image_{i}.png".format(i=i+start_idx)
            >>>         im_i.save(img_folder+'/'+file_name)
            >>>         data_df.loc[i] = [file_name, test_label_arr[i]]
            >>>     return data_df

            >>> test_df = create_df_and_images(path+'t10k-images-idx3-ubyte', path+'t10k-labels-idx1-ubyte', 0, path+'Images')
            >>> train_df = create_df_and_images(path+'train-images-idx3-ubyte', path+'train-labels-idx1-ubyte',len(test_df), path+'Images')
            >>> kept_labels = extra_labels + [target_label]

            >>> instance_id = 0 # Same instance_id for the features and the labels
            >>> for _, row in train_df.iterrows():  
            >>>     if (len(kept_labels)<=1) or (row.label in kept_labels):
            >>>         preprocessor.add_instance_image(instance_id=instance_id, file_path='Images'+os.sep+row.file_name, instances_set=Learning.TRAIN)
            >>>         preprocessor.add_label_class(instance_id=instance_id, label=row.label)
            >>>         instance_id += 1

            >>> for _, row in test_df.iterrows():
            >>>     if (len(kept_labels)<=1) or (row.label in kept_labels):
            >>>         preprocessor.add_instance_image(instance_id=instance_id, file_path='Images'+os.sep+row.file_name, instances_set=Learning.TEST)
            >>>         preprocessor.add_label_class(instance_id=instance_id, label=row.label)
            >>>         instance_id += 1

            >>> preprocessor.to_json(path+'mnist_8vs0.json')

            """
        self.problem_type = _check_parameter(problem_type, ProblemType, "problem_type", "__init__") # Binary Classification, MultiLabel Segmentation, ...
        self.instances_type = _check_parameter(instances_type, InstancesType, "instances_type", "__init__") # Image, tabular, text, temporal, etc..
        self.labels_type = _check_parameter(labels_type, LabelsType, "labels_type", "__init__") # Class, text, mask, contour
        self.instances = {}
        self.labels = {}

    def add_instance_image(self, *, instance_id, file_path, instances_set):
        r"""Add an instance as an image. 

            Parameters
            ----------
            instance_id: ```int```
                A unique identifier for this instance. 

            file_path: ```str```
                The relative file path of the image. 

            instances_set: ```str``` | ```InstancesSet``` | ```None```
                The set (or subset) of the instance (test set, training set, ...) 
                Possible values are defined in the ```InstancesSet``` enum.
            """
        if instance_id in self.instances.keys():
            raise ValueError("The identifier (instance_id) of the instance is aleady used.")
        self.instances[instance_id] = {'file_name': file_path, 'subset':str(instances_set)}

    def add_label_class(self, instance_id, label):
        r"""Add a label as a class. 

            Parameters
            ----------
            id: ```int```
                A unique identifier for this instance. 

            label: ```int```
                The class (label) of this instance. 
        """
        if instance_id in self.labels.keys():
            raise ValueError("The identifier (instance_id) of the instance is aleady used.")
        self.labels[instance_id] = {'instance_id':instance_id, 'label':label}
        
    def add_instances(self, features):
        self.instances.update(features)

    def add_labels(self, labels):
        self.labels.update(labels)
    
    def to_json(self, path): #export ?
        r"""Export the dataset into a JSON file.  

            Parameters
            ----------
            path: ```str```
                The relative path of the JSON file. 
        """
        general_informations = {'problem_type':self.problem_type,
                                'instances_type':self.instances_type, 
                                'labels_type':self.labels_type}
        
        global_dict = {'specificities':general_informations,
                       'instances':self.instances,
                       'labels':self.labels}
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(global_dict, f)
        

