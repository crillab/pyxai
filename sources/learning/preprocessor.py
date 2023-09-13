import pandas

from pyxai.sources.learning.learner import Learner
from pyxai.sources.core.structure.type import TypeFeature, TypeClassification, MethodToBinaryClassification, TypeEncoder, LearnerType
from pyxai import Tools
from pyxai.sources.core.tools.utils import switch_list

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

import os
import numpy
import json
class Preprocessor:
    def __init__(self, dataset, target_feature, learner_type, classification_type=None, to_binary_classification=MethodToBinaryClassification.OneVsRest):
        learner = Learner(learner_type=learner_type)
        self.file = dataset
        self.learner_type = learner_type
        self.classification_type = classification_type
        if self.learner_type == LearnerType.Classification and classification_type is None:
            raise ValueError("Please set the 'classification_type' parameter.")

        self.to_binary_classification = to_binary_classification
        self.data, self.file = learner.parse_data(data=dataset)
        self.n_instances, self.n_features = self.data.shape
        print(self.data.columns)
        self.features_name = list(self.data.columns)
        self.features_type = [None]*self.n_features
        self.numerical_converters = [None]*self.n_features
        self.encoder = [None]*self.n_features
        self.categories = [None]*self.n_features
        self.original_value = [None]*self.n_features
        self.original_types = [str(self.data[feature].dtype) for feature in self.features_name]
        self.already_encoded = [False]*self.n_features
        self.n_bool = 0
        self.target_feature = self.set_target_feature(target_feature)
        self.convert_labels = None
        
    def insert_index(self, index, feature_name, feature_type, numerical_converter, encoder, category, original_value, original_type, already_enc):
        self.features_name.insert(index, feature_name)
        self.features_type.insert(index, feature_type)
        self.numerical_converters.insert(index, numerical_converter)
        self.encoder.insert(index, encoder)
        self.categories.insert(index, category)
        self.original_value.insert(index, original_value)
        self.original_types.insert(index, original_type)
        self.already_encoded.insert(index, already_enc)

    def delete_index(self, index):
        del self.features_name[index]
        del self.features_type[index]
        del self.numerical_converters[index]
        del self.encoder[index]              
        del self.categories[index]              
        del self.original_types[index]
        del self.original_value[index]
        del self.already_encoded[index]

    def switch_indexes(self, index1, index2):
        self.features_name = switch_list(self.features_name, index1, index2)
        self.features_type = switch_list(self.features_type, index1, index2)
        self.numerical_converters = switch_list(self.numerical_converters, index1, index2)
        self.encoder = switch_list(self.encoder, index1, index2)
        self.categories = switch_list(self.categories, index1, index2)
        self.original_types = switch_list(self.original_types, index1, index2)
        self.original_value = switch_list(self.original_value, index1, index2)
        self.already_encoded = switch_list(self.already_encoded, index1, index2)
        
    def set_target_feature(self, feature):
        if feature in self.features_name: 
            self.features_type[self.features_name.index(feature)] = TypeFeature.TARGET
            return self.features_name.index(feature)
        else:
            raise ValueError("The feature called '" + feature + "' is not in the dataset.")
        

    def set_default_type(self, type):
        self.features_type = [type]*self.n_features

    def get_types(self):
        return self.features_type

    def unset_features(self, features):
        for element in features:
            if isinstance(element, str):
                if element in self.features_name:
                    index_element = self.features_name.index(element)
                    if self.features_type[index_element] is not None:
                        raise ValueError("The feature '" + element + "' is already set to "+str(self.features_type[index_element])+".")
                    self.features_type[index_element] = TypeFeature.TO_DELETE
                else:
                    raise ValueError("The feature called '" + element + "' is not in the dataset.")
            elif isinstance(element, int):
                index_element = element
                if self.features_type[index_element] is not None:
                    raise ValueError("The feature '" + index_element + "' is already set to "+str(self.features_type[index_element])+".")
                self.features_type[index_element] = TypeFeature.TO_DELETE
            else:
                raise ValueError("Wrong type for the key " + str(element) + ".")

    def set_categorical_features_already_one_hot_encoded(self, name, features):
        if len(features) == 1:
            element = features[0]
            index_element = self.features_name.index(element)
            self.n_bool += 1
            self.features_type[index_element] = TypeFeature.BINARY
            self.encoder[index_element] = None
            print("The feature " + element + " is boolean! No One Hot Encoding for this features.")
        else:
            for element in features:
                index_element = self.features_name.index(element)
                self.features_type[index_element] = TypeFeature.CATEGORICAL
                self.encoder[index_element] = TypeEncoder.OneHotEncoder
                self.categories[index_element] = name
                self.already_encoded[index_element] = True
                self.original_value[index_element] = (element, features)


    def set_categorical_features(self, columns=None, encoder=TypeEncoder.OneHotEncoder):
        for element in columns:
            if isinstance(element, str):
                if element in self.features_name:
                    index_element = self.features_name.index(element)
                    if self.features_type[index_element] is not None:
                        raise ValueError("The feature '" + element + "' is already set to "+str(self.features_type[index_element])+".")
                    self.features_type[index_element] = TypeFeature.CATEGORICAL
                    self.encoder[index_element] = encoder
                else:
                    raise ValueError("The feature called '" + element + "' is not in the dataset.")
            elif isinstance(element, int):
                index_element = element
                if self.features_type[index_element] is not None:
                    raise ValueError("The feature '" + index_element + "' is already set to "+str(self.features_type[index_element])+".")
                self.features_type[index_element] = TypeFeature.CATEGORICAL
                self.encoder[index_element] = encoder
            else:
                raise ValueError("Wrong type for the key " + str(element) + ".")

    def all_numerical_features(self):
        for element in self.features_name:
            key = self.features_name.index(element)
            if self.target_feature != key and self.features_type[key] != TypeFeature.TO_DELETE:
                self.features_type[key] = TypeFeature.NUMERICAL
                self.numerical_converters[key] = None
                self.encoder[key] = None

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

        #Set the self.features_type and the numerical_converters variables   
        for element in dict_converters.keys():
          if isinstance(element, int):
              if self.features_type[element] is not None:
                  raise ValueError("The feature '" + element + "' is already set to "+str(self.features_type[element])+".")
              self.features_type[element] = TypeFeature.NUMERICAL
              self.numerical_converters[element] = dict_converters[element]
              self.encoder[element] = "CustomizedOrdinalEncoder" if dict_converters[element] is not None else "None"
          else:
              raise ValueError("Wrong type for the key " + str(element) + ".")
          

    def process_target_feature(self):
        # Switch two features to put the target_feature at the end
        
        self.switch_indexes(self.target_feature, -1)
        self.target_feature = -1 #Litte bug: now the target is at the end

        # Move the data
        self.data=self.data[self.features_name]
        
        # Remove instance where the target feature is NaN
        self.target_features_name = self.features_name[-1]
        self.data=self.data.dropna(subset=[self.target_features_name])
        
        # Use the label encoder to encode this feature 
        if self.learner_type == LearnerType.Classification:
            self.encoder[self.target_feature] = "LabelEncoder"
            encoder = LabelEncoder()
            self.data[self.target_features_name] = encoder.fit_transform(self.data[self.target_features_name])
            self.label_encoder_classes = encoder.classes_
        elif self.learner_type == LearnerType.Regression:
            self.encoder[self.target_feature] = "None"
        else:
            raise ValueError("The 'learner_type' parameter is not correct.")
    def process_to_delete(self):
        features_to_delete = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.TO_DELETE]
        indexes_to_delete = []
        for feature in features_to_delete:
            indexes_to_delete.append(self.features_name.index(feature))
            #pandas.concat((indexes_to_delete, pandas.DataFrame(self.features_name.index(feature))),axis=0)

            self.data.drop(feature, inplace=True, axis=1)
            print("Feature deleted: ", feature)    
        
        for index in sorted(indexes_to_delete, reverse=True):
            self.delete_index(index)

    def process_categorical_features(self):     
        features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.CATEGORICAL]
        for feature in features_to_encode:
            index = self.features_name.index(feature)
            if self.already_encoded[index] is True:
                continue
            #print("feature:", feature)
            #print("index:", index)  
            #print("encoder:", self.encoder[index])
            if self.encoder[index] == TypeEncoder.OrdinalEncoder:
                encoder = OrdinalEncoder(dtype=numpy.int)
                data_categorical = self.data[[feature]]      
                #Create a category NaN for missing value in categorical features
                data_categorical = data_categorical.fillna("NaN")
                self.data[[feature]] = encoder.fit_transform(data_categorical)
                self.categories[index] = encoder.categories_
                
            elif self.encoder[index] == TypeEncoder.OneHotEncoder:
                encoder = OneHotEncoder(dtype=numpy.int)
                data_categorical = self.data[[feature]]      
                #Create a category NaN for missing value in categorical features
                #data_categorical = data_categorical.fillna("NaN")
                #print("data:", data_categorical)
                matrix = encoder.fit_transform(data_categorical).toarray()
                names = [element.replace("x0", feature) for element in encoder.get_feature_names_out()]
                original_values = encoder.categories_[0].tolist()
                if len(names) == 2:
                    self.n_bool += 1
                    self.features_type[index] = TypeFeature.BINARY
                    self.encoder[index] = None
                    print("-> The feature " + feature + " is boolean! No One Hot Encoding for this features.") 
                    if isinstance(original_values[0], str):
                        print("-> However, the boolean feature " + feature + " contains strings. A ordinal encoding must be performed.") 
                        encoder = OrdinalEncoder(dtype=numpy.int)
                        data_categorical = data_categorical.fillna("NaN")
                        self.data[[feature]] = encoder.fit_transform(data_categorical)
                        self.categories[index] = encoder.categories_
                        self.encoder[index] = TypeEncoder.OrdinalEncoder
                    continue   
                else:
                    print("One hot encoding new features for " + feature + ": " + str(len(names)))
                
                transformed_df = pandas.DataFrame(matrix, columns=names)
                
                save_features_type = self.features_type[index]
                save_numerical_converter = self.numerical_converters[index]
                save_encoder = self.encoder[index]
                save_category = feature #we put in this variable the original feature :)
                save_original_type = self.original_types[index]
                
                save_already_enc = self.already_encoded[index]
                self.data.drop(feature, inplace=True, axis=1)
                self.delete_index(index)
                
                for i in reversed(range(len(names))):
                    name = names[i]
                    save_original_value = (original_values[i], original_values)
                    self.insert_index(index, name, save_features_type, save_numerical_converter, save_encoder, save_category, save_original_value, save_original_type, save_already_enc)
                    self.data.insert(index, name, transformed_df[name], True)
                #print("index:", index)
                #print("names:", names)
                 
                #print("ff:", self.data.columns)
                #print("features_name:", self.features_name)
                         
            else:
                raise ValueError("Wrong encoder: " + str(self.encoder[index]) + ".")
        
    def process_numerical_features(self):
        features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL and self.numerical_converters[i] is not None]
        converters_to_encode = [self.numerical_converters[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL and self.numerical_converters[i] is not None]

        for i, feature in enumerate(features_to_encode):
            self.data[feature] = self.data[feature].apply(converters_to_encode[i])      
        
        #Remove the NaN value in numerical features:
        features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL]
        self.data[features_to_encode] = self.data[features_to_encode].interpolate(method='linear').fillna(method="bfill")  

    def process(self):
      Tools.verbose("---------------    Converter    ---------------")
      
      if None in self.features_type:
          no_type = [element for i,element in enumerate(self.features_name) if self.features_type[i] is None] 
          raise ValueError("The follow features have no type (please set a type):" + str(no_type))
      
      self.process_target_feature()
    
      self.process_to_delete()

      self.process_categorical_features()
      
      self.process_numerical_features()

      if self.learner_type == LearnerType.Regression:
           self.results = [self.data]
           return self.results
      
      #Lead with Multi or Binary classification
      n_classes = self.data[self.target_features_name].nunique()
      Tools.verbose("Numbers of classes:", n_classes)
      print("Number of boolean features:", self.n_bool)
      if self.classification_type == TypeClassification.MultiClass:
          if n_classes < 3:
              print("Warning: you are chosen TypeClassification.MultiClass but there is "+n_classes+" classes.")
          self.results = [self.data]
          return self.results

      #Binary classification
      if self.classification_type == TypeClassification.BinaryClass:
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

                      #Save the encoding
                      self.convert_labels.append({"Method":str(self.to_binary_classification), 0: others, 1:[int(v1)]})

                      #First replaces
                      data[self.target_features_name] = data[self.target_features_name].replace(v1,new_value_true)
                      for other in others:
                          data[self.target_features_name] = data[self.target_features_name].replace(other,new_value_false)

                      #Replace with 0 and 1
                      data[self.target_features_name] = data[self.target_features_name].replace(new_value_true, 1)
                      data[self.target_features_name] = data[self.target_features_name].replace(new_value_false, 0)
                      self.results.append(data)
                  return self.results
              elif self.to_binary_classification == MethodToBinaryClassification.OneVsOne:  
                  unique_values = self.data[self.target_features_name].unique()
                  for v1 in unique_values: 
                      for v2 in unique_values:
                          if v1 != v2:
                              print("for: ", v1, v2)
                              data = self.data.copy(deep=True)
                              others = [int(v3) for v3 in unique_values if v3 != v1 and v3 != v2]
                              print("others:", others)
                              #Save the encoding
                              self.convert_labels.append({"Method":str(self.to_binary_classification), 0: [int(v2)], 1:[int(v1)]})
                              #delete others
                              for other in others:
                                  data.drop(data[data[self.target_features_name] == other].index, inplace = True)
                              #replace
                              data[self.target_features_name] = data[self.target_features_name].replace(v1, 1)
                              data[self.target_features_name] = data[self.target_features_name].replace(v2, 0)
                              self.results.append(data)
                              last_column = data.iloc[: , -1:]

                              print("last_column:", last_column.nunique())
                  return self.results
              else:
                  raise NotImplementedError()
          else:
              # It is already of the form of a binary class :) 
              self.convert_labels = None

        #while self.data[target_features_name].nunique() > 2:
        #    value_to_remove = self.data[target_features_name][0]
        #    print("The number of classes is reduced by removing some instance with the label " + str#(value_to_remove) + ".")            
        #    self.data.drop(self.data[self.data[target_features_name] == value_to_remove].index, inplace = True)
      self.results = [self.data]
      return self.results

    def export(self, filename, type="csv", output_directory=None):
      for i, dataset in enumerate(self.results):
          self._export(dataset, filename+"_"+str(i), i, type, output_directory)
      Tools.verbose("-----------------------------------------------")
      
    def _export(self, dataset, filename, index, type, output=None):
      # the dataset
      if output is None:
          filename = filename + "." + type
      else:
          filename = output + os.sep + filename + "." + type
      if filename.endswith(".csv"):
          dataset.to_csv(filename, index=False)
          types_filenames = filename.replace(".csv", ".types")
      elif filename.endswith(".xls"):
          dataset.to_xls(filename, index=False)
          types_filenames = filename.replace(".xls", ".types")
      else:
          raise ValueError("The name file of the data_file parameter must be of the type .csv or .xls.")
      
      # the JSON file representing the types of features
      data_type = dict()
      for i, feature in enumerate(self.features_name):
          new_dict = dict()
          new_dict["type:"] = str(self.features_type[i])
          
          new_dict["encoder:"] = str(self.encoder[i])
          if self.encoder[i] == TypeEncoder.OrdinalEncoder:
              new_dict["categories:"] = list(self.categories[i][0]) 
          
          if self.encoder[i] == TypeEncoder.OneHotEncoder:
              new_dict["original_feature:"] = self.categories[i] 
              new_dict["original_values:"] = self.original_value[i] 
              

          #new_dict["original_type:"] = self.original_types[i]
          if self.features_type[i] == TypeFeature.TARGET:
                if self.learner_type == LearnerType.Classification:
                    new_dict["type:"] = str(LearnerType.Classification)
                    new_dict["classes:"] = [str(v) for v in self.label_encoder_classes]
                    if self.convert_labels is not None:
                        new_dict["binary_conversion:"] = self.convert_labels[index]
                elif self.learner_type == LearnerType.Regression:
                    new_dict["type:"] = str(LearnerType.Regression)
                else:
                    raise ValueError("The 'learner_type' parameter is not correct.")
          data_type[feature] = new_dict

      with open(types_filenames, 'w') as outfile:
          json.dump(data_type, outfile, indent=2)

      Tools.verbose("Dataset saved:", filename)
      Tools.verbose("Types saved:", types_filenames)