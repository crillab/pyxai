import pandas

from pyxai.sources.learners.learner import Learner
from pyxai.sources.type import _check_parameter, TypeFeature, ClassificationType, MultiClassToBinaryMethod, EncoderType, ProblemType, LabelsType, InstancesType
from pyxai import Tools
from pyxai.sources.tools.utils import switch_list

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, KBinsDiscretizer

import os
import numpy
import json

# Create an object ```Preprocessor``` giving 
class TabularPreprocessor:
    r"""The ```TabularPreprocessor``` class is used to preprocess tabular datasets.

    The ```TabularPreprocessor``` object give access to methods that allow to:
     - modify the dataset to delete features
     - encode features using an encoder (```OrdinalEncoder```, ```OneHotEncoder```, ```LabelEncoder```) and put the target feature in the last column.
     - rewrite the dataset in a new CSV file and save in a JSON file the type of features (numerical or categorical)
    """
    
    def __init__(self, 
                 dataset, 
                 target_feature, 
                 problem_type, 
                 classification_type=None, 
                 to_binary_classification=MultiClassToBinaryMethod.OneVsOne, 
                 discretization=None):
        r"""Initialise a ```TabularPreprocessor``` object with a dataset. 



            Parameters
            ----------
            dataset: ```str``` | ```pandas.DataFrame``` | ```NoneData```
                The dataset to use, either as a path to a csv, json or excel file or as a pandas DataFrame.

            target_feature: ```str```
                The feature name of the target feature.

            problem_type: ```str``` | ```ProblemType```
                The type of problem (classification, regression, ...)
                Possible values are defined in the ```ProblemType``` enum.
                    
            classification_type: ```str``` | ```ClassificationType``` (optional, default=```None```)
                The type of classification (BinaryClass or MultiClass) to produce.  
                Possible values are defined in the ```ClassificationType``` enum.

            to_binary_classification: ```str``` | ```MultiClassToBinaryMethod``` (optional, default=```MultiClassToBinaryMethod.OneVsOne```)
                The method used to encode a multi-classes dataset into a binary class dataset (OneVsOne or OneVsRest)
                Possible values are defined in the ```MultiClassToBinaryMethod``` enum.
                    
            discretization: ```int``` (optional, default=```None```)
                If this parameter is None, no discretization is used. 
                Else, it is the number of bins to produce. 
                This discretization method uses a [```KBinsDiscretizer```](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) to transform numerical features into categorical features (with a direct encoding).   
                
            Returns
            -------
            ```TabularPreprocessor```:
               A ```TabularPreprocessor``` object

            Examples
            --------
            >>> import pandas
            >>> from pyxai import Learning, Tools

            >>> data = pandas.read_csv(Tools.Options.dataset, names=["sepal length", "sepal width", "petal length", "petal width", "Iris Plants"])
            >>> preprocessor = Learning.TabularPreprocessor(data, target_feature="Iris Plants", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

            >>> preprocessor.all_numerical_features()
            >>> preprocessor.process()

            >>> dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
            >>> preprocessor.export(dataset_name, output_directory=Tools.Options.output)

            Examples
            --------
            >>> import datetime
            >>> from pyxai import Learning, Tools

            >>> preprocessor = Learning.TabularPreprocessor(Tools.Options.dataset, target_feature="Type", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS, to_binary_classification=Learning.ONE_VS_REST)

            >>> preprocessor.unset_features(["Address", "Suburb", "SellerG"])

            >>> preprocessor.set_categorical_features(features=["Method", "CouncilArea", "Regionname"])

            >>> preprocessor.set_numerical_features({
            >>>   "Postcode": lambda d: int(d),
            >>>   "Rooms": None, 
            >>>   "Price": None,
            >>>   "Date": lambda d: datetime.date(int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])).toordinal(), 
            >>>   "Distance": None,
            >>>   "Bedroom2": None,
            >>>   "Bathroom": None,
            >>>   "Car": None,
            >>>   "Landsize": None,
            >>>   "BuildingArea": None,
            >>>   "YearBuilt": None,
            >>>   "Lattitude": None,
            >>>   "Longtitude": None,
            >>>   "Propertycount": None
            >>>   })

            >>> preprocessor.process()

            >>> dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
            >>> preprocessor.export(dataset_name, output_directory=Tools.Options.output)

            
            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/preprocessor/)
            - More examples are in the [```pyxai/examples/Converters```](https://github.com/crillab/pyxai/tree/main/pyxai/examples/Converters) directory in the source code. 
        """

        learner = Learner(problem_type=problem_type, instances_type=InstancesType.Tabular)
        self.file = dataset
        self.problem_type = _check_parameter(problem_type, ProblemType, "problem_type", "__init__")
        self.classification_type = classification_type
        if self.problem_type == ProblemType.Classification and classification_type is None:
            raise ValueError("Please set the 'classification_type' parameter.")

        self.to_binary_classification = to_binary_classification
        self.data, self.file = learner._parse_data(data=dataset)
        self.n_instances, self.n_features = self.data.shape
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
        self.extra_names = []

        self.discretization = discretization
        self.discretization_n_bins = [None]*self.n_features
        self.discretization_bin_edges = [None]*self.n_features
        


    def insert_index(self, index, feature_name, feature_type, numerical_converter, encoder, category, original_value, original_type, already_enc):
        self.features_name.insert(index, feature_name)
        self.features_type.insert(index, feature_type)
        self.numerical_converters.insert(index, numerical_converter)
        self.encoder.insert(index, encoder)
        self.categories.insert(index, category)
        self.discretization_n_bins.insert(index, None)
        self.discretization_bin_edges.insert(index, None)
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
        del self.discretization_n_bins[index]
        del self.discretization_bin_edges[index]

    def switch_indexes(self, index1, index2):
        self.features_name = switch_list(self.features_name, index1, index2)
        self.features_type = switch_list(self.features_type, index1, index2)
        self.numerical_converters = switch_list(self.numerical_converters, index1, index2)
        self.encoder = switch_list(self.encoder, index1, index2)
        self.categories = switch_list(self.categories, index1, index2)
        self.original_types = switch_list(self.original_types, index1, index2)
        self.original_value = switch_list(self.original_value, index1, index2)
        self.already_encoded = switch_list(self.already_encoded, index1, index2)
        self.discretization_n_bins = switch_list(self.discretization_n_bins, index1, index2)
        self.discretization_bin_edges = switch_list(self.discretization_bin_edges, index1, index2)

    def set_target_feature(self, feature):
        if feature in self.features_name: 
            self.features_type[self.features_name.index(feature)] = TypeFeature.Target
            return self.features_name.index(feature)
        else:
            raise ValueError("The feature called '" + feature + "' is not in the dataset.")
        

    def set_default_type(self, type):
        for i in range(len(self.features_type)):
            if self.features_type[i] is None:
                self.features_type[i]=type

    def get_types(self):
        return self.features_type

    def unset_features(self, features):
        r"""Delete the features given in parameter.

            Parameters
            ----------
            features: ```list[str | int]```
                List of features to remove. 
                In this list, a feature can be represented either by a ```str``` (a feature name), or a ```int``` (a feature index).  

            Examples
            --------
            >>> from pyxai import Learning, Tools

            >>> preprocessor = Learning.TabularPreprocessor(Tools.Options.dataset, target_feature="Type", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS, to_binary_classification=Learning.ONE_VS_REST)
            >>> preprocessor.unset_features(["Address", "Suburb", "SellerG"])
            >>> preprocessor.process()
            """
        for element in features:
            if isinstance(element, str):
                if element in self.features_name:
                    index_element = self.features_name.index(element)
                    if self.features_type[index_element] is not None:
                        raise ValueError("The feature '" + element + "' is already set to "+str(self.features_type[index_element])+".")
                    self.features_type[index_element] = TypeFeature.To_delete
                else:
                    raise ValueError("The feature called '" + element + "' is not in the dataset.")
            elif isinstance(element, int):
                index_element = element
                if self.features_type[index_element] is not None:
                    raise ValueError("The feature '" + index_element + "' is already set to "+str(self.features_type[index_element])+".")
                self.features_type[index_element] = TypeFeature.To_delete
            else:
                raise ValueError("Wrong type for the key " + str(element) + ".")
            

    def set_categorical_features_already_one_hot_encoded(self, name, features):
        r"""Identify a set of binary features coming from a one hot encoding as a categorical feature.

            Parameters
            ----------
            name: ```str```
                The name of the resulting categorical feature.

            features: ```list[str]```
                Feature names in the dataset corresponding to binary features where each feature represents a category.

            Examples
            --------
            >>> from pyxai import Learning, Tools

            >>> preprocessor = Learning.TabularPreprocessor(
            >>>     "../dataset/compas.csv", 
            >>>     target_feature="Two_yr_Recidivism", 
            >>>     problem_type=Learning.CLASSIFICATION, 
            >>>     classification_type=Learning.BINARY_CLASS)

            >>> preprocessor.set_categorical_features_already_one_hot_encoded("score_factor", ["score_factor"])
            >>> preprocessor.set_categorical_features_already_one_hot_encoded("Age_Above_FourtyFive", ["Age_Above_FourtyFive"])
            >>> preprocessor.set_categorical_features_already_one_hot_encoded("Age_Below_TwentyFive", ["Age_Below_TwentyFive"])
            >>> preprocessor.set_categorical_features_already_one_hot_encoded("Ethnic", ["African_American", "Asian", "Hispanic", "Native_American", "Other"])
            >>> preprocessor.set_categorical_features_already_one_hot_encoded("Female", ["Female"])
            >>> preprocessor.set_categorical_features_already_one_hot_encoded("Misdemeanor", ["Misdemeanor"])

            >>> preprocessor.set_numerical_features({
            >>>     "Number_of_Priors": None
            >>> })

            >>> preprocessor.process()
            >>> preprocessor.export("compas", output_directory="../dataset")
            """
        if len(features) == 1:
            element = features[0]
            index_element = self.features_name.index(element)
            self.n_bool += 1
            self.features_type[index_element] = TypeFeature.Binary
            self.encoder[index_element] = None
            print("The feature " + element + " is boolean! No One Hot Encoding for this features.")
        else:
            for element in features:
                index_element = self.features_name.index(element)
                self.features_type[index_element] = TypeFeature.Categorical
                self.encoder[index_element] = EncoderType.OneHotEncoder
                self.categories[index_element] = name
                self.already_encoded[index_element] = True
                self.original_value[index_element] = (element, features)


    def set_categorical_features(self, features=None, encoder=EncoderType.OneHotEncoder):
        r"""Encode the categorical features given in the ```columns``` parameter with an encoder given in the ```encoder``` parameter.
        
            Parameters
            ----------
            features: ```list[str | int]```
                The features to encode.  
                These features can be given either as indexes (respecting the order of the given dataset) or directly via strings corresponding to their names.
            
            encoder: ```str``` | ```EncoderType``` (optional, default=```EncoderType.OneHotEncoder```)
                The type of encoder (one-hot or ordinal)
                Possible values are defined in the ```EncoderType``` enum.
            
            Examples
            --------
            >>> from pyxai import Learning, Tools

            >>> preprocessor = Learning.TabularPreprocessor(Tools.Options.dataset, target_feature="Type", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS, to_binary_classification=Learning.ONE_VS_REST)
            >>> preprocessor.set_categorical_features(features=["Method", "CouncilArea", "Regionname"])
            >>> preprocessor.process()
            """
        
        for element in features:
            if isinstance(element, str):
                if element in self.features_name:
                    index_element = self.features_name.index(element)
                    if self.features_type[index_element] is not None:
                        raise ValueError("The feature '" + element + "' is already set to "+str(self.features_type[index_element])+".")
                    self.features_type[index_element] = TypeFeature.Categorical
                    self.encoder[index_element] = encoder
                else:
                    raise ValueError("The feature called '" + element + "' is not in the dataset.")
            elif isinstance(element, int):
                index_element = element
                if self.features_type[index_element] is not None:
                    raise ValueError("The feature '" + index_element + "' is already set to "+str(self.features_type[index_element])+".")
                self.features_type[index_element] = TypeFeature.Categorical
                self.encoder[index_element] = encoder
            else:
                raise ValueError("Wrong type for the key " + str(element) + ".")

    def all_numerical_features(self):
        r"""Identify all features as numerical.

            Examples
            --------
            >>> from pyxai import Learning, Tools

            >>> preprocessor = Learning.TabularPreprocessor(Tools.Options.dataset, target_feature="1636", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)
            >>> preprocessor.all_numerical_features()

            >>> preprocessor.process()
            >>> dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
            >>> preprocessor.export(dataset_name, output_directory=Tools.Options.output)
            """
        for element in self.features_name:
            key = self.features_name.index(element)
            if self.target_feature != key and self.features_type[key] != TypeFeature.To_delete:
                self.features_type[key] = TypeFeature.Numerical
                self.numerical_converters[key] = None
                self.encoder[key] = None

    def set_numerical_features(self, features_dict):
        r"""Select and encode the numerical features present in the dictionary ```features_dict```.

            Parameters
            ----------
            features_dict: ```dict[str | int] -> Lambda | None```
                Python dictionary where keys are numerical features and values are lambda functions representing data encodings. 
                The keys must be represented either by ```str``` (feature names) or ```int``` (feature indexes). 
                Note that a lambda function set to ```None``` means that no encoding must be done.
            
            Examples
            --------
            >>> import datetime
            >>> from pyxai import Learning, Tools

            >>> preprocessor = Learning.TabularPreprocessor("../dataset/melb.csv", 
                                                target_feature="Type", 
                                                problem_type=Learning.CLASSIFICATION, 
                                                classification_type=Learning.MULTI_CLASS)
            >>> preprocessor.unset_features(["Address", "Suburb", "SellerG"])
            >>> preprocessor.set_categorical_features(features=["Method", "CouncilArea", "Regionname"])
            >>> preprocessor.set_numerical_features({
                "Postcode": lambda d: int(d),
                "Date": lambda d: datetime.date(int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])).toordinal(), 
                "Distance": None, "Bedroom2": None, "Bathroom": None,
                "Car": None, "Landsize": None, "BuildingArea": None, "YearBuilt": None,
                "Lattitude": None, "Longtitude": None, "Propertycount": None,
                "Rooms": None, "Price": None
            })
            >>> preprocessor.process()
            >>> preprocessor.export("melb", output_directory="../dataset")                                    
            """
        
        #Convert the integer keys into string features
        new_dict_converters = dict() 
        for element in features_dict.keys():
            if isinstance(element, str):
                if element in self.features_name: 
                    new_dict_converters[self.features_name.index(element)] = features_dict[element]
                else:
                    raise ValueError("The feature called '" + element + "' is not in the dataset.")
            elif isinstance(element, int):
                new_dict_converters[element] = features_dict[element]
            else:
              raise ValueError("Wrong type for the key " + str(element) + ".")
        features_dict = new_dict_converters

        #Set the self.features_type and the numerical_converters variables   
        for element in features_dict.keys():
          if isinstance(element, int):
              if self.features_type[element] is not None:
                  raise ValueError("The feature '" + element + "' is already set to "+str(self.features_type[element])+".")
              self.features_type[element] = TypeFeature.Numerical
              self.numerical_converters[element] = features_dict[element]
              self.encoder[element] = "CustomizedOrdinalEncoder" if features_dict[element] is not None else "None"
          else:
              raise ValueError("Wrong type for the key " + str(element) + ".")
          

    def process_target_feature(self):
        # Switch two features to put the target_feature at the end
        
        self.switch_indexes(self.target_feature, -1)
        #self.target_feature = -1 #Litte bug: now the target is at the end

        # Move the data
        self.data=self.data[self.features_name]
        
        # Remove instance where the target feature is NaN
        self.target_features_name = self.features_name[-1]
        self.data=self.data.dropna(subset=[self.target_features_name])
        
        # Use the label encoder to encode this feature 
        if self.problem_type == ProblemType.Classification:
            self.encoder[self.target_feature] = "LabelEncoder"
            encoder = LabelEncoder()
            self.data[self.target_features_name] = encoder.fit_transform(self.data[self.target_features_name])
            self.label_encoder_classes = encoder.classes_
        elif self.problem_type == ProblemType.Regression:
            self.encoder[self.target_feature] = "None"
        else:
            raise ValueError("The 'problem_type' parameter is not correct.")
            
    def process_to_delete(self):
        features_to_delete = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.To_delete]
        indexes_to_delete = []
        for feature in features_to_delete:
            indexes_to_delete.append(self.features_name.index(feature))
            #pandas.concat((indexes_to_delete, pandas.DataFrame(self.features_name.index(feature))),axis=0)

            self.data.drop(feature, inplace=True, axis=1)
            print("Feature deleted: ", feature)    
        
        for index in sorted(indexes_to_delete, reverse=True):
            self.delete_index(index)

    
    def process_categorical_features(self):     
        features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.Categorical]
        for feature in features_to_encode:
            index = self.features_name.index(feature)
            if self.already_encoded[index] is True:
                continue
            if self.encoder[index] == EncoderType.OrdinalEncoder:
                encoder = OrdinalEncoder(dtype=int)
                data_categorical = self.data[[feature]]      
                #Create a category NaN for missing value in categorical features
                data_categorical = data_categorical.fillna("NaN")
                self.data[[feature]] = encoder.fit_transform(data_categorical)
                self.categories[index] = encoder.categories_
                
            elif self.encoder[index] == EncoderType.OneHotEncoder:
                encoder = OneHotEncoder(dtype=int)
                data_categorical = self.data[[feature]]      
                #Create a category NaN for missing value in categorical features
                #data_categorical = data_categorical.fillna("NaN")
                matrix = encoder.fit_transform(data_categorical).toarray()
                names = [element.replace("x0", feature) for element in encoder.get_feature_names_out()]
                original_values = encoder.categories_[0].tolist()
                if len(names) == 2:
                    self.n_bool += 1
                    self.features_type[index] = TypeFeature.Binary
                    self.encoder[index] = None
                    print("-> The feature " + feature + " is boolean! No One Hot Encoding for this features.") 
                    if isinstance(original_values[0], str):
                        print("-> However, the boolean feature " + feature + " contains strings. A ordinal encoding must be performed.") 
                        encoder = OrdinalEncoder(dtype=int)
                        data_categorical = data_categorical.fillna("NaN")
                        self.data[[feature]] = encoder.fit_transform(data_categorical)
                        self.categories[index] = encoder.categories_
                        self.encoder[index] = EncoderType.OrdinalEncoder
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
                         
            else:
                raise ValueError("Wrong encoder: " + str(self.encoder[index]) + ".")
        
    def process_numerical_features(self):
        features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.Numerical and self.numerical_converters[i] is not None]
        converters_to_encode = [self.numerical_converters[i] for i, t in enumerate(self.features_type) if t == TypeFeature.Numerical and self.numerical_converters[i] is not None]
        for i, feature in enumerate(features_to_encode):
            self.data[feature] = self.data[feature].apply(converters_to_encode[i])   
            
        
        #Remove the NaN value in numerical features:
        features_to_encode = [self.features_name[i] for i, t in enumerate(self.features_type) if t == TypeFeature.Numerical]
        #for i, feature in enumerate(features_to_encode):
        #    self.data[feature] = pandas.to_numeric(self.data[feature])
        self.data[features_to_encode] = self.data[features_to_encode].interpolate(method='linear').bfill()  
        if self.discretization is not None:
            print("Discretization of numerical features:")
            discretizer = KBinsDiscretizer(n_bins=self.discretization, encode='ordinal').fit(self.data[features_to_encode])
            self.data[features_to_encode] = discretizer.transform(self.data[features_to_encode])
            for feature in features_to_encode:
                index = self.features_name.index(feature)
                self.encoder[index] = EncoderType.DiscretizerEncoder
                self.discretization_n_bins[index] = discretizer.n_bins_[features_to_encode.index(feature)]
                self.discretization_bin_edges[index] = discretizer.bin_edges_[features_to_encode.index(feature)]

                print("discretizer.bin_edges_ for feature " + feature + ":", discretizer.bin_edges_[features_to_encode.index(feature)])
                print("discretizer.n_bins_ for feature " + feature + ":", discretizer.n_bins_[features_to_encode.index(feature)])

                

        
    def process(self):
      r"""  Transform the dataset.

            Applies all encodings previously defined by other methods of this class.
            - Put the target feature in the last column 
            - Delete the feature to remove
            - Encode categorical and numerical features
            - Convert a multi classes dataset into a binary class if needed

            Returns
            -------
            ```pandas.DataFrame```:  
               The new dataset. 
               
            Examples
            --------
            >>> from pyxai import Learning, Tools

            >>> preprocessor = Learning.TabularPreprocessor(Tools.Options.dataset, target_feature="Type", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS, to_binary_classification=Learning.ONE_VS_REST)
            >>> preprocessor.set_categorical_features(features=["Method", "CouncilArea", "Regionname"])
            >>> preprocessor.process()
            """
      
      Tools.verbose("---------------    Converter    ---------------")
      
      if None in self.features_type:
          no_type = [element for i,element in enumerate(self.features_name) if self.features_type[i] is None] 
          raise ValueError("The follow features have no type (please set a type):" + str(no_type))
      
      self.process_target_feature()
    
      self.process_to_delete()

      self.process_categorical_features()
      
      self.process_numerical_features()

      if self.problem_type == ProblemType.Regression:
           self.results = [self.data]
           return self.results
      
      #Lead with Multi or Binary classification
      n_classes = self.data[self.target_features_name].nunique()
      Tools.verbose("Numbers of classes:", n_classes)
      print("Number of boolean features:", self.n_bool)
      if self.classification_type == ClassificationType.MultiClass:
          if n_classes < 3:
              print("Warning: you are chosen TypeClassification.MultiClass but there is "+n_classes+" classes.")
          self.results = [self.data]
          return self.results

      #Binary classification
      if self.classification_type == ClassificationType.BinaryClass:
          if n_classes > 2:
              print("Warning: conversion from MultiClass to BinaryClass: the current dataset will be convert into several new datasets with the " + str(self.to_binary_classification) + " method.")
              self.results = []
              self.convert_labels = []
              if self.to_binary_classification == MultiClassToBinaryMethod.OneVsRest:
                  unique_values = self.data[self.target_features_name].unique()
                  new_value_true = max(unique_values)+1
                  new_value_false = max(unique_values)+2
                  for v1 in unique_values:
                      print("MethodToBinaryClassification.OneVsRest: ", str(v1) + "_vs_others")
                              
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
                      self.extra_names.append(str(v1) + "_vs_others")
                  return self.results
              elif self.to_binary_classification == MultiClassToBinaryMethod.OneVsOne:  
                  unique_values = list(self.data[self.target_features_name].unique())
                  unique_values.sort()
                  already_done = []
                  nValues = [len(self.data[self.data[self.target_features_name]==v]) for v in unique_values]
                  
                  for v1 in unique_values: 
                      for v2 in unique_values:
                          if v1 != v2:
                              data = self.data.copy(deep=True)
                              if set([v1, v2]) in already_done:
                                  continue
                              already_done.append(set([v1, v2]))

                              others = [int(v3) for v3 in unique_values if v3 != v1 and v3 != v2]
                              #Save the encoding
                              self.convert_labels.append({"Method":str(self.to_binary_classification), 0: [int(v2)], 1:[int(v1)]})
                              #delete others
                              for other in others:
                                  data.drop(data[data[self.target_features_name] == other].index, inplace = True)
                              
                              #replace
                              
                              data[self.target_features_name] = data[self.target_features_name].replace(v1, "v1")
                              data[self.target_features_name] = data[self.target_features_name].replace(v2, "v0")
                              data[self.target_features_name] = data[self.target_features_name].replace("v1", 1)
                              data[self.target_features_name] = data[self.target_features_name].replace("v0", 0)
                              
                              nV1 = len(data[data[self.target_features_name]==1])
                              nV2 = len(data[data[self.target_features_name]==0])
                              print("MethodToBinaryClassification.OneVsOne: ", str(nV1) + " - " + str(nV2))  
                              self.results.append(data)
                              self.extra_names.append(str(v1) + "_vs_" + str(v2))
                              #last_column = data.iloc[: , -1:]
                              #print("last_column:", last_column.nunique())
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
      r"""  Export the dataset that has been transformed.

            This function creates two files, the first one represents the new dataset and the second one is a JSON file containing information about the transformations made on the data.
            In the case where a multiclass classification problem is converted to a set of binary classification problems, several pairs of files are produced. 

            Parameters
            ----------
            filename: ```str```
                The filename of new files.

            type: ```str``` (optional, default=```csv```)
                The type of the new dataset (“csv” or “xls”).

            output_directory: ```str```
                The target directory for the new files (this directory must exist).
               
            Examples
            --------

            >>> from pyxai import Learning

            >>> preprocessor = Learning.TabularPreprocessor("../dataset/iris.csv", 
                                                target_feature="Species", 
                                                problem_type=Learning.CLASSIFICATION, 
                                                classification_type=Learning.BINARY_CLASS)

            >>> preprocessor.all_numerical_features()

            >>> preprocessor.process() 
            >>> preprocessor.export("iris", output_directory="../dataset")

            ---------------    Converter    ---------------
            Numbers of classes: 3
            Number of boolean features: 0
            Warning: conversion from MultiClass to BinaryClass: the current dataset will be convert into several new datasets with the OneVsRest method.
            Dataset saved: ../dataset/iris_0.csv
            Types saved: ../dataset/iris_0.types
            Dataset saved: ../dataset/iris_1.csv
            Types saved: ../dataset/iris_1.types
            Dataset saved: ../dataset/iris_2.csv
            Types saved: ../dataset/iris_2.types
            -----------------------------------------------
            """
      for i, dataset in enumerate(self.results):
          p_filename = filename+"_"+self.extra_names[i] if len(self.extra_names) != 0 else filename
          self._export(dataset, p_filename, i, type, output_directory)
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
          if self.encoder[i] == EncoderType.OrdinalEncoder:
              new_dict["categories:"] = list(self.categories[i][0]) 
          
          if self.encoder[i] == EncoderType.OneHotEncoder:
              new_dict["original_feature:"] = self.categories[i] 
              new_dict["original_values:"] = self.original_value[i] 

          if self.encoder[i] == EncoderType.DiscretizerEncoder:
              new_dict["discretization_n_bins:"] = str(self.discretization_n_bins[i])
              new_dict["discretization_bin_edges:"] = [str(value).strip() for value in self.discretization_bin_edges[i]]

          #new_dict["original_type:"] = self.original_types[i]
          if self.features_type[i] == TypeFeature.Target:
                if self.problem_type == ProblemType.Classification:
                    new_dict["type:"] = str(ProblemType.Classification)
                    new_dict["classes:"] = [str(v) for v in self.label_encoder_classes]
                    if self.convert_labels is not None:
                        new_dict["binary_conversion:"] = self.convert_labels[index]
                elif self.problem_type == ProblemType.Regression:
                    new_dict["type:"] = str(ProblemType.Regression)
                else:
                    raise ValueError("The 'problem_type' parameter is not correct.")
          data_type[feature] = new_dict

      with open(types_filenames, 'w') as outfile:
          json.dump(data_type, outfile, indent=2)

      Tools.verbose("Dataset saved:", filename)
      Tools.verbose("Types saved:", types_filenames)