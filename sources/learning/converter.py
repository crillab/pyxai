from pyxai.sources.learning.Learner import Learner
from pyxai.sources.core.structure.type import TypeFeature, TypeLearner, TypeClassification, MethodToBinaryClassification
from pyxai import Tools
from pyxai.sources.core.tools.utils import switch_list

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import numpy
import json
class Converter:
    def __init__(self, dataset, target_feature, classification_type, to_binary_classification=MethodToBinaryClassification.OneVsRest):
        learner = Learner()
        self.data, self.file = learner.parse_data(data=dataset)
        self.n_instances, self.n_features = self.data.shape
        self.features_name = list(self.data.columns)
        self.features_type = [None]*self.n_features
        self.dict_converters_numerical = None
        self.classification_type = classification_type
        self.to_binary_classification = to_binary_classification
        self.encoder = [None]*self.n_features
        self.original_types = [str(self.data[feature].dtype) for feature in self.features_name]

        self.target_feature = self.set_target_feature(target_feature)
        
      

        
    def set_target_feature(self, feature):
        if feature in self.features_name: 
            self.features_type[self.features_name.index(feature)] = TypeFeature.TARGET
            return self.features_name.index(feature)
        else:
            raise ValueError("The feature called '" + feature + "' is not in the dataset.")
    
     
    def process_target_feature(self):
        # Switch two features to put the target_feature at the end
        self.encoder[self.target_feature] = "LabelEncoder" 
        self.features_name = switch_list(self.features_name, self.target_feature, -1)
        self.original_types = switch_list(self.original_types, self.target_feature, -1)
        self.features_type = switch_list(self.features_type, self.target_feature, -1)
        self.encoder = switch_list(self.encoder, self.target_feature, -1)
        
        # if the last is in keys, we have to change the key
        if len(self.features_type)-1 in self.dict_converters_numerical.keys():
            self.dict_converters_numerical[self.target_feature] = self.dict_converters_numerical[len(self.features_type)-1]
            del self.dict_converters_numerical[len(self.features_type)-1]

        print(self.dict_converters_numerical)
        
        self.data=self.data[self.features_name]
        
        
        # Remove instance where the target feature is NaN
        self.target_features_name = self.features_name[-1]
        self.data=self.data.dropna(subset=[self.target_features_name])
        
        # Use the label encoder to encode this feature 
        encoder = LabelEncoder()
        self.data[self.target_features_name] = encoder.fit_transform(self.data[self.target_features_name])
        self.label_encoder_classes = encoder.classes_
        

    def set_default_type(self, type):
        self.features_type = [type]*self.n_features

    def get_types(self):
        return self.features_type

    def set_categorical_features(self, columns_id=None, columns_name=None):
        if columns_id is not None:
            for id in columns_id:
                if self.features_type[id] is not None:
                    raise ValueError("The feature '" + id + "' is already set to "+str(self.features_type[id])+".")
                self.features_type[id] = TypeFeature.CATEGORICAL
        if columns_name is not None:
            for name in columns_name:
                if name in self.features_name: 
                    if self.features_type[self.features_name.index(name)] is not None:
                        raise ValueError("The feature '" + name + "' is already set to "+str(self.features_type[self.features_name.index(name)])+".")
                    self.features_type[self.features_name.index(name)] = TypeFeature.CATEGORICAL
                else:
                    raise ValueError("The feature called '" + name + "' is not in the dataset.")


    def set_numerical_features(self, dict_converters):
        #Convert the integer keys into string features
        new_dict_converters = dict() 
        for element in dict_converters.keys():
            if isinstance(element, str):
                if element in self.features_name: 
                    new_dict_converters[self.features_name.index(element)] = dict_converters[element]
                else:
                    raise ValueError("The feature called '" + element + "' is not in the dataset.")
            elif isinstance(element, int):
                new_dict_converters[element] = dict_converters[element]
            else:
              raise ValueError("Wrong type for the key " + str(element) + ".")
        dict_converters = new_dict_converters

        #Set the self.features_type variable   
        for element in dict_converters.keys():
          if isinstance(element, int):
              if self.features_type[element] is not None:
                  raise ValueError("The feature '" + element + "' is already set to "+str(self.features_type[element])+".")
              self.features_type[element] = TypeFeature.NUMERICAL
          else:
              raise ValueError("Wrong type for the key " + str(element) + ".")
          
        #Save the global variable
        self.dict_converters_numerical = dict_converters
        



    def process(self):
        
      if None in self.features_type:
          no_type = [element for i,element in enumerate(self.features_name) if self.features_type[i] is None] 
          raise ValueError("The follow features have no type (please set a type):" + str(no_type))
      
      self.process_target_feature()
      print("self.features_type", self.features_type)
      print("self.original_types:", self.original_types)
      
      #process categorical features
      encoder = OrdinalEncoder(dtype=numpy.int)
      features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.CATEGORICAL]
      data_categorical = self.data[features_to_encode]      
      #Create a category NaN for missing value in categorical features
      data_categorical = data_categorical.fillna("NaN")
      self.data[features_to_encode] = encoder.fit_transform(data_categorical)
      for i, t in enumerate(self.features_type):
          if t == TypeFeature.CATEGORICAL:
              self.encoder[i] = "OrdinalEncoder"

      #process numerical features
      features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL and self.dict_converters_numerical[i] is not None]
      converters_to_encode = [self.dict_converters_numerical[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL and self.dict_converters_numerical[i] is not None]
      for i, feature in enumerate(features_to_encode):
          self.data[feature] = self.data[feature].apply(converters_to_encode[i])      
      
      for i, t in enumerate(self.features_type):
          if t == TypeFeature.NUMERICAL:
              if self.dict_converters_numerical[i] is not None:
                  self.encoder[i] = "CustomizedOrdinalEncoder"
              else:
                  self.encoder[i] = "None"

      #Remove the NaN value in numerical features:
      features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL]
      self.data[features_to_encode] = self.data[features_to_encode].interpolate(method='linear').fillna(method="bfill")     

      #Lead with Multi or Binary classification
      n_classes = self.data[self.target_features_name].nunique()
      print("n_classes:", n_classes)
      if self.classification_type == TypeClassification.MultiClass:
          if n_classes < 3:
              print("Warning: you are chosen TypeClassification.MultiClass but there is "+n_classes+" classes.")
          self.results = [self.data]
          return self.results

      #Binary classification
      if n_classes > 2:
          print("Warning: conversion from MultiClass to BinaryClass: the current dataset will be convert into several new datasets with the " + str(self.to_binary_classification) + " method.")
          self.results = []
          self.convert_labels = []
          if self.to_binary_classification == MethodToBinaryClassification.OneVsRest:
              unique_values = self.data[self.target_features_name].unique()
              new_value_true = max(unique_values)+1
              new_value_false = max(unique_values)+2
              for v1 in unique_values:
                  data = self.data.copy(deep=True)
                  others = [int(v2) for v2 in unique_values if v2 != v1]
                  self.convert_labels.append({0: others, 1:[int(v1)]})

                  data[self.target_features_name] = data[self.target_features_name].replace(v1,new_value_true)
                  for other in others:
                      data[self.target_features_name] = data[self.target_features_name].replace(other,new_value_false)

                  data[self.target_features_name] = data[self.target_features_name].replace(new_value_true, 1)
                  data[self.target_features_name] = data[self.target_features_name].replace(new_value_false, 0)
                  self.results.append(data)
              return self.results
          elif self.to_binary_classification == MethodToBinaryClassification.OneVsOne:            
              raise NotImplementedError()
          else:
              raise NotImplementedError()

        #while self.data[target_features_name].nunique() > 2:
        #    value_to_remove = self.data[target_features_name][0]
        #    print("The number of classes is reduced by removing some instance with the label " + str#(value_to_remove) + ".")            
        #    self.data.drop(self.data[self.data[target_features_name] == value_to_remove].index, inplace = True)

    
      return self.data

    def export(self, filename, type="csv"):
      for i, dataset in enumerate(self.results):
          self._export(dataset, filename+"_"+str(i), i, type)

    def _export(self, dataset, filename, index, type):
      # the dataset
      filename = filename + "." + type
      if filename.endswith(".csv"):
          dataset.to_csv(filename, index=False)
          types_filenames = filename.replace(".csv", ".types")
      elif filename.endswith(".xls"):
          dataset.to_csv(filename, index=False)
          types_filenames = filename.replace(".xls", ".types")
      else:
          raise ValueError("The name file of the data_file parameter must be of the type .csv or .xls.")
      
      # the JSON file representing the types of features
      data_type = dict()
      for i, feature in enumerate(self.features_name):
          new_dict = dict()
          new_dict["type:"] = str(self.features_type[i])
          new_dict["encoder:"] = self.encoder[i]
          new_dict["original_type:"] = self.original_types[i]
          if self.features_type[i] == TypeFeature.TARGET:
              new_dict["classes:"] = list(self.label_encoder_classes)
              new_dict["binary_conversion:"] = self.convert_labels[index]
          data_type[feature] = new_dict

      with open(types_filenames, 'w') as outfile:
          json.dump(data_type, outfile, indent=2)

      Tools.verbose("Dataset saved:", filename)
      Tools.verbose("Types saved:", types_filenames)