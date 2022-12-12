from pyxai.sources.learning.Learner import Learner
from pyxai.sources.core.structure.type import TypeFeature
from pyxai import Tools

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import numpy
import json
class Converter:
    def __init__(self, dataset):
        learner = Learner()
        self.data, self.file = learner.parse_data(data=dataset)
        self.n_instances, self.n_features = self.data.shape
        self.features_type = [None]*self.n_features
        self.dict_converters_numerical = None
        self.encoder = [None]*self.n_features
        self.original_types = [str(self.data[feature].dtype) for feature in self.features_name()]
    
    def set_default_type(self, type):
        self.features_type = [type]*self.n_features

    def get_types(self):
        return self.features_type

    def features_name(self):
        return tuple(self.data.columns)

    def set_categorical_features(self, columns_id=None, columns_name=None):
        if columns_id is not None:
            for id in columns_id:
                self.features_type[id] = TypeFeature.CATEGORICAL
        if columns_name is not None:
            features_name = self.features_name()
            for name in columns_name:
                if name in features_name: 
                    self.features_type[features_name.index(name)] = TypeFeature.CATEGORICAL
                else:
                    raise ValueError("The feature called '" + name + "' is not in the dataset.")


    def set_numerical_features(self, dict_converters):
        features_name = self.features_name()
        #Convert the integer keys into string features
        new_dict_converters = dict() 
        for element in dict_converters.keys():
            if isinstance(element, str):
                if element in features_name: 
                    new_dict_converters[features_name.index(element)] = dict_converters[element]
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
              self.features_type[element] = TypeFeature.NUMERICAL
          else:
              raise ValueError("Wrong type for the key " + str(element) + ".")
          
        #Save the global variable
        self.dict_converters_numerical = dict_converters

    def process(self):
      features_name = self.features_name()
      if None in self.features_type:
          no_type = [element for i,element in enumerate(features_name) if self.features_type[i] is None] 
          raise ValueError("The follow features have no type (please set a type):" + str(no_type))
      
      #process categorical features
      encoder = OrdinalEncoder(dtype=numpy.int)
      features_to_encode = [features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.CATEGORICAL]
      data_categorical = self.data[features_to_encode]      
      #Create a category NaN for missing value in categorical features
      data_categorical = data_categorical.fillna("NaN")
      self.data[features_to_encode] = encoder.fit_transform(data_categorical)
      for i, t in enumerate(self.features_type):
          if t == TypeFeature.CATEGORICAL:
              self.encoder[i] = "OrdinalEncoder"

      #process numerical features
      features_to_encode = [features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL and self.dict_converters_numerical[i] is not None]
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
      features_to_encode = [features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.NUMERICAL]
      self.data[features_to_encode] = self.data[features_to_encode].interpolate(method='linear').fillna(method="bfill")     
      return self.data

    def export(self, filename):
      # the dataset
      if filename.endswith(".csv"):
          self.data.to_csv(filename, index=False)
          types_filenames = filename.replace(".csv", ".types")
      elif filename.endswith(".xls"):
          self.data.to_csv(filename, index=False)
          types_filenames = filename.replace(".xls", ".types")
      else:
          raise ValueError("The name file of the data_file parameter must be of the type .csv or .xls.")
      
      # the JSON file representing the types of features
      data_type = dict()
      features_name = self.features_name()
      for i, feature in enumerate(features_name):
          new_dict = dict()
          new_dict["type:"] = str(self.features_type[i])
          new_dict["encoder:"] = self.encoder[i]
          new_dict["original_type:"] = self.original_types[i]
          data_type[feature] = new_dict

      json_string = json.dumps(data_type)
      with open(types_filenames, 'w') as outfile:
          json.dump(json_string, outfile)

      Tools.verbose("Dataset saved:", filename)
      Tools.verbose("Types saved:", types_filenames)