import os
import numpy
import pandas

from copy import deepcopy
import json
import dill
import base64
import random
from shutil import rmtree
from collections import OrderedDict
from typing import Iterable
from sklearn.model_selection import LeaveOneGroupOut, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from collections.abc import Callable

import inspect

from pyxai import Tools

from pyxai.sources.data_structures.linearModel import LinearModel
from pyxai.sources.data_structures.boostedTrees import BoostedTrees
from pyxai.sources.data_structures.decisionTree import DecisionTree
from pyxai.sources.data_structures.randomForest import RandomForest
from pyxai.sources.data_structures.baseModel import BaseModel
from pyxai.sources.type import _check_parameter, SplittingMethod, ProblemType, ModelType, InstancesType, LabelsType, InstancesSet, TypeFeature
from pyxai.sources.tools.utils import flatten, Metric
from pyxai.sources.learners.learner_information import LearnerInformation

        
class NoneData: pass

class Learner:
    r"""
    This class is a wrapper around the different libraries used to create the models (sklearn, xgboost, lightgbm, ...).
    It stores a dataset, evaluates it using different models and provides some tools to get instances, predictions and metrics.
    """
    
    def __init__(self, data: str|pandas.DataFrame|NoneData = NoneData, *, 
                    problem_type: str|ProblemType, 
                    model_type: str|ModelType|None = None, 
                    instances_type: str|InstancesType|None = None,
                    labels_type: str|LabelsType|None = None,
                    get_item_function: Callable|None = None,
                    instances_directory: str|None = None,
                    labels_directory: str|None = None):
        r"""In the case of a csv or excel file, the last column is considered as the label column and all other columns are considered as features. 
            In the case of a json file, the dataset should be in a specific format (see documentation). 

            Parameters
            ----------
            data: ```str``` | ```pandas.DataFrame``` | ```NoneData```
                The dataset to use, either as a path to a csv, json or excel file or as a pandas DataFrame.

            problem_type: ```str``` | ```ProblemType```
                The type of problem (classification, regression, ...)
                Possible values are defined in the ```ProblemType``` enum.
            
            model_type: ```str``` | ```ModelType``` (optional, default=```None```)
                The type of model (linear, tree-based, neural network, ...)
                Can be ```None``` and put in the evaluation method. 
                Possible values are defined in the ```ModelType``` enum.

            instances_type: ```str``` | ```InstancesType``` (optional, default=```None```)
                The type of instances (image, tabular, text, temporal, ...)
                Possible values are defined in the ```InstancesType``` enum.
                When set to None, if data is a csv and get_item_function is None then it is set to "tabular"
                    
            labels_type: ```str``` | ```LabelsType``` (optional, default=```None```)
                The type of labels (class, text, mask, contours, ...)
                Possible values are defined in the ```LabelsType``` enum.
                When set to None, if problem_type is "classification" then it is set to "classes", if problem_type is "regression" then it is set to "continuous-values"

            get_item_function: ```Callable``` (optional, default=```None```)
                A function to get an instance from the dataset. This function is used to get an instance in the right format for the model and the explainer.
                If the dataset is a pandas DataFrame and the instances are tabular, this function is not necessary and can be set to None. 
                In other cases, this function should be defined by the user. It should take as input a row of the dataframe and return the corresponding instance in the right format for the model and the explainer.

            instances_directory: ```str``` (optional, default=```None```)
                The directory where the instances are stored (only for a JSON dataset). 
                This parameter is used to extend the path of instances in the dataframe when the instances are of type image and the dataset is given as a json file.
            
            labels_directory: ```str``` (optional, default=```None```)
                The directory where the labels are stored (only for a JSON dataset). 
                This parameter is used to extend the paths of labels in the dataframe when the labels are of type masks and the dataset is given as a json file.
                Warning: NOT YET IMPLEMENTED


            Examples
            --------
            >>> from pyxai import Learning, Tools
            >>> learner = Learning.Scikitlearn(Tools.Options.dataset, problem_type=Learning.CLASSIFICATION)
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
            >>> instance, prediction = learner.get_instances(n=1)
            

            Examples
            --------
            >>> from pyxai import Learning, Tools
            >>> learner = Learning.Xgboost(Tools.Options.dataset, problem_type=Learning.CLASSIFICATION)
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, splitting_parameters={'test_size':0.2}, model_parameters={'max_depth':6, 'base_score':0.5})
            
            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/learning/generating/)
        """


        if problem_type is None:
            raise ValueError("Please set the parameter 'problem_type' to 'classification' or 'regression'.")
        self.problem_type = _check_parameter(problem_type, ProblemType, "problem_type", "__init__") # Type of the problem (classification, regression, ...)
        self.model_type = _check_parameter(model_type, ModelType, "model_type", "__init__") # Type of the model (linear, tree-based, neural network, ...)
        self.instances_type = _check_parameter(instances_type, InstancesType, "instances_type", "__init__") # Type of instances: image, tabular, text, temporal, ...
        self.labels_type = _check_parameter(labels_type, LabelsType, "labels_type", "__init__") # Type of labels: class, text, mask, contours, ...
        self.get_item_function = get_item_function
        self.instances_directory = instances_directory
        self.labels_directory = labels_directory

        self.instances_dataframe, self.labels_dataframe = None, None
        self.n_labels = None  # Number of unique labels in the dataset, None for Regression
        self.classes = None

        self.splitting_method = None  # Method used for splitting (e.g., cross-validation)
        self.n_categorical = None  # Number of categorical features
        self.n_numerical = None  # Number of numerical features
        self.learner_information_lst = []  # List to store additional information about the learner
        
        # The data returned is either a Dataframe (from a CSV or a excel file) or a Dict (from a json file) or None
        _, self.dataset_path = self._parse_data(data)

        if self.instances_dataframe is not None:
            self.upper_bounds = tuple(self.instances_dataframe.max(axis=0))
            self.lower_bounds = tuple(self.instances_dataframe.min(axis=0))

        if self.labels_dataframe is not None:
            if self.labels_type == LabelsType.Classes:
                self.classes = self.labels_dataframe.label.unique()
                self.n_labels = len(self.labels_dataframe.label.unique())
            elif self.labels_type == LabelsType.Masks:
                raise NotImplementedError('Masks labels_type implementation in progress')
                #self.labels_df['mask_path'] = img_folder+'/'+self.labels_df.file_name
                #self.labels_dataframe = self.labels_dataframe.drop(columns=['file_name'])
            elif self.labels_type == LabelsType.Contours:
                raise NotImplementedError('Contours labels_type implementation in progress')
                cv2_adaptation = lambda seg_lst: [numpy.array([seg['x'], seg['y']], dtype=int).T for seg in seg_lst]
                self.labels_dataframe.segmentation = self.labels_dataframe.segmentation.apply(cv2_adaptation)

        if self.get_item_function is None:
            if self.instances_type == InstancesType.Tabular: self.get_item_function = lambda x: x
            elif self.instances_type is None: raise ValueError('Please indicate either instance_type or get_item_function')
            else: raise ValueError('Please indicate get_item_function')
        else:
            self.get_item_function = get_item_function

        if self.instances_dataframe is not None:
            first_instance = self.get_item_function(self.instances_dataframe.iloc[0]) if self.instances_dataframe is not None else None
            if isinstance(first_instance, pandas.Series):
                self.feature_names = list(first_instance.index)
            else:
                self.feature_names = None
            self.n_features = len(first_instance)
            self.n_instances = len(self.instances_dataframe)
        else:
            self.feature_names, self.n_features, self.n_instances = None, None, None
        if self.feature_names is not None and "subset" in self.feature_names:
            self.feature_names.remove('subset') # TODO Rename subset in order to make sure that the dataframe can't have a feature with this name
            self.n_features += -1


        Tools.verbose("--------------   Information   ---------------")
        Tools.verbose("Problem type:", self.problem_type)
        if model_type: Tools.verbose("Model type:", self.model_type)
        Tools.verbose("Instances type:", self.instances_type)
        Tools.verbose("Labels type:", self.labels_type)
        
        Tools.verbose("")
        Tools.verbose("Dataset path:", self.dataset_path)
        if self.n_features is not None:
            Tools.verbose("nFeatures (nAttributes, with the labels):", self.n_features)
        if self.n_instances is not None:
            Tools.verbose("nInstances (nObservations):", self.n_instances)
        if self.n_labels is not None:
            Tools.verbose("nLabels:", self.n_labels)
        

    
    def _parse_data_from_dict(self, data_dict):
        if self.problem_type is None: self.problem_type = data_dict['specificities']['problem_type']
        if self.instances_type is None: self.instances_type = data_dict['specificities']['instances_type']
        if self.labels_type is None: self.labels_type = data_dict['specificities']['labels_type']

        self.instances_dataframe = pandas.DataFrame.from_dict(data_dict['instances'], orient='index')
        if self.instances_type == InstancesType.Image:
            self.instances_dataframe.index = self.instances_dataframe.index.astype(int)
            if self.instances_directory is not None:
                self.instances_dataframe['path'] = self.instances_directory+os.sep+self.instances_dataframe.file_name
            else:
                self.instances_dataframe['path'] = './'+self.instances_dataframe.file_name
            self.instances_dataframe = self.instances_dataframe.drop(columns=['file_name'])
        else:
            raise ValueError('instances_type must be specified as "image"')
            
        self.labels_dataframe = pandas.DataFrame.from_dict(data_dict['labels'], orient='index')
        self.labels_dataframe.index = self.labels_dataframe.index.astype(int)


        
    def _parse_data_from_dataframe(self, data_dataframe: pandas.DataFrame):
        if self.problem_type is None: raise ValueError("Please set the 'problem_type' parameter.")

        if self.instances_type is None:
            if self.get_item_function is None:
                self.instances_type = "tabular"
            else:
                raise ValueError("When using a custom get_item_function, please set the 'instances_type' parameter.")
        elif self.instances_type != InstancesType.Tabular:
            raise ValueError("In the case of a dataframe, 'instances_type' should be 'tabular'.")

        if self.labels_type is None:
            if self.problem_type==ProblemType.Classification:
                self.labels_type="classes"
            elif self.problem_type==ProblemType.Regression:
                self.labels_type="continuous-values"
            else:
                raise ValueError("If the problem_type is not classification or regression, please set the 'labels_type' parameter.")
                    
        self.instances_dataframe = data_dataframe.iloc[:,:-1]
       
        self.labels_dataframe = data_dataframe.iloc[:,-1:].copy()
        self.labels_dataframe.columns = ["label"]
    
    
    def get_details(self):
        r"""Get some details about the learner and the models. This information is available in the ```LearnerInformation``` class.

            Returns
            -------
            ```list``` of ```dict```:
               A list with a python Dictionnary for each model. 

            Examples
            --------
            >>> from pyxai import Learning
            >>> learner = Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
            >>> models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.DT)
            >>> for id, models in enumerate(models):
            >>>     metrics = learner.get_details()[id]["metrics"]           
                
        """
        return [learner_information.to_dict() for learner_information in self.learner_information_lst]

    
    def get_raw_models(self):
        r"""Get the raw models from the librairy used to create the models (sklearn, xgboost, lightgbm, ...)
            
            Returns
            -------
            ```list``` of (```DecisionTreeClassifier``` | ```RandomForestClassifier``` | ```XGBClassifier``` | ```XGBRegressor``` | ```LGBMRegressor```):
               A list of raw models. 
            
            Examples
            --------
            >>> from pyxai import Learning
            >>> learner = Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
            >>> models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.DT)
            >>> for id, models in enumerate(models):
            >>>     raw_model = learner.get_raw_models()[id]           
            
        """
        return [learner_information.raw_model for learner_information in self.learner_information_lst]

    
    def _parse_data(self, data: pandas.DataFrame|str|NoneData = NoneData, 
                   skip=None, n=None) -> tuple[pandas.DataFrame|dict|None, str|None]:
        """
        Return the data according to their type (dataframe or json dictionnary) and the path of the dataset
        """
        if data is NoneData:
            return None, None
        elif isinstance(data, str):
            if os.path.isfile(data):
                if data.endswith(".csv") or data.endswith(".xlsx"):
                    data_dataframe = pandas.read_csv(data, skiprows=skip, nrows=n).copy()
                    self._parse_data_from_dataframe(data_dataframe)
                    return  data_dataframe, data
                elif data.endswith(".json"):
                    with open(data) as f: data_dict = json.load(f)
                    self._parse_data_from_dict(data_dict)
                    return data_dict, data
                else:
                    raise ValueError("If data is a string, it must be a csv, a json or or excel file with either the .csv, .json or .xlsx extension.")
            raise FileNotFoundError("The file " + data + " is not found.")

        elif isinstance(data, pandas.core.frame.DataFrame):
            self._parse_data_from_dataframe(data)
            return data, "path unknown" 

        elif data is None:
            raise ValueError("The 'data' parameter is set but None (can be optional, but not None). If you are using 'Tools.Options.dataset' as parameter, check if the '-dataset=<dataset_file>' option is present in your command.")

        raise ValueError("The data parameter must be a string representing a csv, a json, a excel file or a pandas.core.frame.DataFrame object.")

    
    def compute_metrics(self, labels, predictions):
        r"""Returns the values of some metrics (accuracy, recall, f1_score, specificity, ...) according to the evaluation method.

            Parameters
            ----------
            labels: ```int``` | ```float```
                The real values (labels) in the dataset.     
            
            predictions: ```int``` | ```float```
                The prediction of a model.     
            
            Returns
            -------
            ```dict```:
                A Python dictionary where the keys depend on the ```ProblemType```.
                See the page of the ```Metrics``` class for more information.  
            
            Examples
            --------
            >>> from pyxai import Learning
            >>> learner = Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
            >>> labels = [1,1,1,1,1,0,0,0,0,0]
            >>> predictions = [1,1,1,1,1,0,0,0,0,0]
            >>> metrics = learner.compute_metrics(labels, predictions)

        """
    
        if self.problem_type == ProblemType.Classification:
            if self.n_labels == 2: # binary class case
                # TP, TN, FP and FN is available only for the two binary class case. 
                return Metric.compute_metrics_binary_classification(labels, predictions)
            else: #multiclass case
                return Metric.compute_metrics_multi_classification(labels, predictions)
        elif self.problem_type == ProblemType.Regression:
            return Metric.compute_metrics_regression(labels, predictions)
        else:
            raise ValueError("problem_type is unknown:", str(self.problem_type))

            
    def evaluate(self, *, splitting_method, model_type, model_parameters={}, splitting_parameters={}):
        r"""Runs an experimental protocol using the train-test split technique based on cross-validation methods. 
            It makes the train-test split according to the cross validation method using Scikit-learn then it executes the fit and predict operation of the chosen ML classifier (Scikit-learn, XGBoost, LightGBM) as many times as necessary.
            
            Parameters
            ----------

            splitting_method: ```str``` | ```SplittingMethod```
                The splitting method used for the evalution (hold-out, k-folds, ...)
                Possible values are defined in the ```SplittingMethod``` enum.

            model_type: ```str``` | ```ModelType``` 
                The type of model (linear, tree-based, neural network, ...)
                Possible values are defined in the ```ModelType``` enum.

            model_parameters: ```dict``` (optional, default={})
                Parameters to pass to the learner according to the used librairy (sklearn, xgboost, ...)
                For example, for a ```RandomForestClassifier```, we can set ```model_parameters``` to ```{"n_estimators":50, "max_depth":4}```.

            splitting_parameters: ```dict``` (optional, default={})
                Parameters to pass to the splitting method (hold-out, k-folds, ...)
                For example, for the ```Learning.LEAVE_ONE_GROUP_OUT``` method, we can set ```splitting_parameters``` to ```{'n_models':2, 'random_state':0}```.

            Returns
            -------
            ```DecisionTree``` | ```RandomForest``` | ```BoostedTrees``` | ```BoostedTreesRegression```:
                The PyXai model created.
            
            ```list``` of (```DecisionTree``` | ```RandomForest``` | ```BoostedTrees``` | ```BoostedTreesRegression```):
                A list of PyXai models when several models were created with the splitting method.

            Notes
            -----
            The default values of ```model_parameters``` and ```splitting_parameters``` are the empty Python dictionnaries. 
            As a result, the random seeds are set to `None` by default for the model and the splitting method. 
            This makes the evaluation non-deterministic by default. 
            To make it deterministic, you must add to model_parameters or/and splitting_parameters the key/value ```'random_state':0```.   
            
            Examples
            --------
            >>> from pyxai import Learning
            >>> learner = Learning.LightGBM("tests/datasets/dermatology.csv", problem_type=Learning.REGRESSION)
            >>> models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.BT, splitting_parameters={'random_state':0}, model_parameters={'learning_rate':0.3, 'n_estimators':5, 'random_state':0})

            
        """
        self.splitting_method = _check_parameter(splitting_method, SplittingMethod, "splitting_method", "evaluate")
        self.model_type = _check_parameter(model_type, ModelType, "model_type", "evaluate")

        #if "random_state" not in splitting_parameters.keys(): # 12/02/2026 TODO remove for using the default parameters from the evaluation method
        #    splitting_parameters["random_state"] = None
        
        Tools.verbose("---------------   Model creation, fitting and evaluation  ---------------")
        Tools.verbose("Splitting method:", self.splitting_method)
        Tools.verbose("Problem type:", self.problem_type)
        Tools.verbose("Models type:", self.model_type)
        if self.splitting_method == SplittingMethod.Predefined:
            self._predefined_evaluation(splitting_parameters=splitting_parameters, model_parameters=model_parameters)
        elif self.splitting_method == SplittingMethod.HoldOut:
            self._hold_out_evaluation(splitting_parameters=splitting_parameters, model_parameters=model_parameters)
        elif self.splitting_method == SplittingMethod.LeaveOneGroupOut:
            self._leave_one_group_out_evaluation(splitting_parameters=splitting_parameters, model_parameters=model_parameters)
        elif self.splitting_method == SplittingMethod.KFolds:
            self._k_folds_evaluation(splitting_parameters=splitting_parameters, model_parameters=model_parameters)
        else:
            raise NotImplementedError("Implementation in progress for {} splitting method".format(self.splitting_method))

        Tools.verbose("---------   Evaluation Information   ---------")
        for i, result in enumerate(self.learner_information_lst):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose("Metrics:")
            for key in result.metrics.keys():
                if key == "sklearn_confusion_matrix" and len(result.metrics[key])>20:
                    continue
                Tools.verbose("   " + key + ": " +str(result.metrics[key]))
                
            Tools.verbose("Number of Training instances:", len(result.train_index))
            Tools.verbose("Number of Testing instances:", len(result.test_index))
            Tools.verbose()

        Tools.verbose("---------------   Explainer   ----------------")
        pyxai_structures_lst = self.convert_model()

        for i, pyxai_structure in enumerate(pyxai_structures_lst):
            Tools.verbose("For the split number " + str(i) + ":")
            Tools.verbose(pyxai_structure)

        return pyxai_structures_lst if len(pyxai_structures_lst) != 1 else pyxai_structures_lst[0]

    def _predefined_evaluation(self, *, splitting_parameters, model_parameters):
        """
        For each instance the user have predifined if it's part of the Train or Test subset by creating a subset column
        """
        self.learner_information_lst.clear()
        assert self.instances_dataframe is not None, "You have to put the dataset in the class parameters."
        assert "subset" in self.instances_dataframe.columns, "You should define Train and Test instances in a new column called subset"
        if 'n_models' in splitting_parameters.keys():
            assert splitting_parameters['n_models'] == 1, "For predefined_evaluation the number of models should be one"
        for key in splitting_parameters.keys():
            if key != "n_models" and splitting_parameters[key]!=None:
                print('{} defined in splitting_parameters but not used'.format(key)) # TO DO transform into a warning

        instances_train = self.instances_dataframe[self.instances_dataframe.subset.str.lower()=='train']
        instances_test = self.instances_dataframe[self.instances_dataframe.subset.str.lower()=='test']
        if "instance_id" not in self.labels_dataframe.columns:
            self.labels_dataframe["instance_id"] = self.labels_dataframe.index
        labels_train = self.labels_dataframe[self.labels_dataframe.instance_id.isin(instances_train.index)].label
        labels_test = self.labels_dataframe[self.labels_dataframe.instance_id.isin(instances_test.index)].label
        train_index = instances_train.index
        test_index = instances_test.index
        instances_train = instances_train.drop(columns='subset')
        instances_test = instances_test.drop(columns='subset')

        model, metrics, extras = self.fit_and_predict(instances_train, instances_test, labels_train, labels_test, model_parameters)
        self.learner_information_lst.append(self._create_learner_information(model, train_index, test_index, None, metrics, extras))
        return self

    
    def _hold_out_evaluation(self, *, splitting_parameters, model_parameters):
        if 'n_models' in splitting_parameters.keys():
            assert splitting_parameters['n_models'] == 1, "For hold_out_evaluation the number of models should be one"
        SPLITTING_KEYS = list(inspect.signature(train_test_split).parameters.keys())[1:]
        for key in splitting_parameters.keys():
            if key not in SPLITTING_KEYS + ["n_models"]:
                print('{} defined in splitting_parameters but not used'.format(key)) # TODO transform into a warning
        splitting_parameters = {key:splitting_parameters[key] for key in splitting_parameters if key in SPLITTING_KEYS}

        self.learner_information_lst.clear()
        assert self.instances_dataframe is not None, "You have to put the dataset in the class parameters."

        instances_train, instances_test, labels_train, labels_test, train_index, test_index = train_test_split(self.instances_dataframe, self.labels_dataframe,
                                                                                                               self.instances_dataframe.index, **splitting_parameters)
        model, metrics, extras = self.fit_and_predict(instances_train, instances_test, labels_train.label, labels_test.label, model_parameters)
        self.learner_information_lst.append(self._create_learner_information(model, train_index, test_index, None, metrics, extras))
        return self


    def _k_folds_evaluation(self, *, splitting_parameters, model_parameters):
        if 'n_models' in splitting_parameters.keys():
            splitting_parameters['n_splits'] = splitting_parameters['n_models']
            del splitting_parameters['n_models']
        if 'shuffle' not in splitting_parameters.keys():
            splitting_parameters['shuffle'] = True # Why not keeping sklearn default value ? (False) # TODO make a choice between pyxai and sklearn default value
        if "n_splits" in splitting_parameters.keys():
            assert splitting_parameters['n_splits'] > 1, "This k_folds_evaluation() expects at least 2 parts. For just one part, please use hold_out_evaluation() or predefined_evaluation()"
        SPLITTING_KEYS = list(inspect.signature(KFold).parameters.keys()) # TODO refactoring
        for key in splitting_parameters.keys():
            if key not in SPLITTING_KEYS + ["n_models"]:
                print('{} defined in splitting_parameters but not used'.format(key)) # TODO transform into a warning
        splitting_parameters = {key:splitting_parameters[key] for key in splitting_parameters if key in SPLITTING_KEYS}

        self.learner_information_lst.clear()
        assert self.instances_dataframe is not None, "You have to put the dataset in the class parameters."
        
        cross_validator = KFold(**splitting_parameters)

        for train_index, test_index in cross_validator.split(self.instances_dataframe):
            # Select good observations for each of the 'n_trees' experiments.
            instances_train = self.instances_dataframe.iloc[train_index]
            labels_train = self.labels_dataframe.label.iloc[train_index]
            instances_test = self.instances_dataframe.iloc[test_index]
            labels_test = self.labels_dataframe.label.iloc[test_index]

            model, metrics, extras = self.fit_and_predict(instances_train, instances_test, labels_train, labels_test, model_parameters)
            
            # Save key informations
            self.learner_information_lst.append(self._create_learner_information(model, train_index, test_index, None, metrics, extras))
        return self


    def _leave_one_group_out_evaluation(self, *, splitting_parameters, model_parameters):
        assert self.instances_dataframe is not None, "You have to put the dataset in the class parameters."
        if 'n_models' in splitting_parameters.keys():
            assert splitting_parameters['n_models'] > 1, "leave_one_group_out_evaluation() expects at least 2 parts. For just one part, please use hold_out_evaluation() or predefined_evaluation()"
            n_models = splitting_parameters['n_models']
        else:
            n_models = 5
        if 'random_state' not in splitting_parameters.keys():
            splitting_parameters['random_state'] = None
        for key in splitting_parameters.keys():
            if key not in ["random_state", "n_models"]:
                print('{} defined in splitting_parameters but not used'.format(key)) # TODO transform into a warning
        self.learner_information_lst.clear()

        # spliting
        quotient, remainder = (self.n_instances // n_models, self.n_instances % n_models)
        groups = flatten([quotient * [i] for i in range(1, n_models + 1)]) + [i for i in range(1, remainder + 1)]
        random.Random(splitting_parameters["random_state"]).shuffle(groups)
        cross_validator = LeaveOneGroupOut()

        for train_index, test_index in cross_validator.split(self.instances_dataframe, self.labels_dataframe, groups):
            # Select good observations for each of the 'n_trees' experiments.
            instances_train = self.instances_dataframe.iloc[train_index]
            labels_train = self.labels_dataframe.label.iloc[train_index]
            instances_test = self.instances_dataframe.iloc[test_index]
            labels_test = self.labels_dataframe.label.iloc[test_index]

            # solving
            model, metrics, extras = self.fit_and_predict(instances_train, instances_test, labels_train, labels_test, model_parameters)
            
            # Save key informations
            self.learner_information_lst.append(self._create_learner_information(model, train_index, test_index, groups, metrics, extras))
        return self

    def get_instances(self,
                      model=None,
                      indexes=InstancesSet.All,
                      *,
                      dataset=None,
                      n=None,
                      is_correct=None,
                      subset_predicted_classes=None,
                      subset_true_classes=None,
                      save_directory=None,
                      instances_id=None,
                      seed=0,
                      train_indexes=None,
                      test_indexes=None,
                      details=False) -> list[tuple[pandas.Series, int|float|str]] | list[dict] | tuple[None, None] | tuple[pandas.Series, int|float|str] | dict :
        r"""Return couples (instance, prediction) from the dataset with the given prediction done by the model.

            Parameters
            ----------
            model: ```DecisionTree``` | ```RandomForest``` | ```BoostedTrees``` | ```BoostedTreesRegression``` (optional, default=None):
                A PyXAI model.

            indexes: ```str``` | ```InstancesSet``` (optional, default=```InstancesSet.All```)
                The type of selected instances (all, from the training instances, from the testing instances, ...)
                Possible values are defined in the ```InstancesSet``` enum.
                Can also be a ```str``` representing a file contening specific indexes.

            dataset: ```str``` | ```pandas.DataFrame``` | ```NoneData``` (optional, default=```None```)
                The dataset to use, either as a path to a csv, json or excel file or as a pandas DataFrame.
                Can be None if the dataset is already loaded. This parameter is useful only if the dataset is not already loaded.

            n: ```int``` (optional, default=```None```)
                The desired number of instances (```None``` for all).
                    
            is_correct: ```True``` | ```False``` (optional, default=```None```)
                Only available if a model is given 
                    - ```None```: All instances (no filter).
                    - ```True```: Only correctly classified instances by the model.
                    - ```False```: Only misclassified instances by the model. 

            subset_predicted_classes: ```list[int]``` (optional, default=```None```)  
                - ```None```: All instances (no filter).
                - ```list[int]```: List of classes for the desired instances considering the model prediction.

            subset_true_classes: ```list[int]``` (optional, default=```None```)  
                - ```None```: All instances (no filter).
                - ```list[int]```: List of classes for the desired instances, taking into account the true labels.

            save_directory: ```str``` (optional, default=```None```)
                Save the instance indexes in a file in the directory given by this parameter 

            instances_id : ```str``` (optional, default=```None```)
                To add an identifier in the name of the saved file with the ```save_directory``` parameter or useful to load instances using the ```indexes``` parameter.

            seed: ```int``` | ```None``` (optional, default=```0```)
                Set to ```None``` to obtain fully random instances. Default value is ```0```. Set the seed to an ```Integer``` to shuffle the result with this seed.

            train_indexes: ```list[int]``` (optional, default=```None```)
                List of training indexes to select the instances from subset of indexes. 
            
            test_indexes: ```list[int]``` (optional, default=```None```)
                List of testing indexes to select the instances from subset of indexes. 
            
            details: ```True``` | ```False``` (optional, default=```False```)
                Set to ```True``` to obtain a python list of instances where each instance is in the form of Python dictionnaries with the keys: “instance”, “prediction”, “label” and “index”.

                
            Returns
            -------
            ```list[tuple(pandas.Series, int | float | str)]``` | ```list[dict]```:
                Return couples (instance, prediction) the selected instances with the given prediction done by the model.

            ```tuple(pandas.Series, int | float | str)``` | ```dict```:
                Note that when the number of instances requested is only 1 (```n=1```), the method returns ony one tuple.
                Python dictionnaries are returned when the parameter ```details``` is set to ```True```
            
            Examples
            --------
            >>> from pyxai import Learning
            >>> learner = Learning.Scikitlearn(dataset, problem_type='regression')
            >>> model = learner.evaluate(splitting_method='hold-out', model_type='linear-ridge', splitting_parameters={'random_state':0}, model_parameters={})
            >>> instances = learner.get_instances(model, indexes='train-in-priority', n=5, seed=72)

            Examples
            --------
            >>> from pyxai import Learning
            >>> learner = Learning.Scikitlearn(dataset, problem_type='classification')
            >>> model = learner.evaluate(splitting_method='hold-out', model_type='linear-ridge', splitting_parameters={'random_state':0}, model_parameters={})
            >>> instances = learner.get_instances(model, indexes='test', n=5, subset_predicted_classes=['Iris-versicolor', 'Iris-virginica'])

            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/learning/generating/)
        """

        # 1: Check parameters and get the associated solver
        Tools.verbose("---------------   Instances   ----------------")
        assert isinstance(indexes, (InstancesSet, str)), "Bad value in the parameter 'indexes'"
        if subset_predicted_classes is not None:
            assert self.problem_type==ProblemType.Classification, "subset_predicted_classes parameter is only available for classification problems"
        if subset_true_classes is not None:
            assert self.problem_type==ProblemType.Classification, "subset_true_classes parameter is only available for classification problems"
        if is_correct is not None:
            assert self.problem_type==ProblemType.Classification, "is_correct parameter is only available for classification problems"
        
        raw_model, learner_information = self._get_raw_model(model, indexes, is_correct)
            
        # 2: Get the good indexes
        if isinstance(dataset, pandas.DataFrame) and 'subset' in dataset.columns:
            if train_indexes is None:
                train_indexes = dataset[dataset.subset.str.lower()=='train'].index
            if test_indexes is None:
                test_indexes = dataset[dataset.subset.str.lower()=='test'].index
        possible_indexes = self._get_possible_indexes(indexes, instances_id, learner_information, train_indexes, test_indexes)
        
        # 3: Select only the data needed
        possible_indexes, data_instances, data_labels = self._get_data(dataset, possible_indexes, subset_true_classes)
        original_indexes = self._build_original_indexes(possible_indexes, seed)
        
        # 4: Select instances according to data and possible_indexes
        instances: list[tuple] = []
        instances_indexes: list[int] = []
        instances_details: list[dict] = []
        if raw_model is None:
            for j in original_indexes:
                instances.append((data_instances.iloc[j], None))
                instances_indexes.append(possible_indexes[j])
                instances_details.append({"instance": data_instances.iloc[j], "prediction": None, "label": None, "index":possible_indexes[j]})
                if isinstance(n, int) and len(instances) >= n:
                    break
        else:
            is_filling_conditions = lambda x, label: True if ( (is_correct is None) or ( (is_correct is True) and (x==label) ) or ( (is_correct is False) and (x != label) ) ) and \
                            ( (subset_predicted_classes is None) or (x in subset_predicted_classes) ) else False
            instances_to_predict = data_instances.loc[original_indexes]
            corresponding_labels = data_labels.label.loc[original_indexes]
            if isinstance(n, int) and n<len(original_indexes) and ( (is_correct is not None) or (subset_predicted_classes is not None) ):
                for i in original_indexes:
                    instance_i = instances_to_predict.loc[i]
                    pred = self.predict(raw_model, pandas.DataFrame([instance_i]))[0]
                    label = corresponding_labels.loc[i]
                    if is_filling_conditions(pred, label):
                        instances.append((instance_i, pred))
                        instances_indexes.append(i)
                        instances_details.append({"instance": instance_i, "prediction": pred, "label": label, "index":i})
                        if len(instances) >= n:
                            break
            else:
                if isinstance(n, int) and n<len(original_indexes):
                    instances_to_predict = instances_to_predict[:n]
                    corresponding_labels = corresponding_labels[:n]
                    original_indexes = original_indexes[:n]
                all_predictions_solver = pandas.DataFrame(self.predict(raw_model, instances_to_predict), index=original_indexes)
                if (is_correct is not None) or (subset_predicted_classes is not None):
                    check_conditions = [is_filling_conditions(all_predictions_solver.loc[i,0], corresponding_labels.loc[i]) for i in original_indexes]
                    all_predictions_solver = all_predictions_solver[check_conditions]
                    instances_to_predict = instances_to_predict[check_conditions]
                    corresponding_labels = corresponding_labels[check_conditions]
                    original_indexes = [original_indexes[i] for i in range(len(original_indexes)) if check_conditions[i]]
                for i in original_indexes:
                    instances.append((instances_to_predict.loc[i], all_predictions_solver.loc[i,0]))
                    instances_indexes.append(i)
                    instances_details.append({"instance": instances_to_predict.loc[i], "prediction": all_predictions_solver.loc[i,0], "label": corresponding_labels.loc[i], "index":i})
            if len(instances) == 0:
                    raise ValueError("No instances are corresponding to the selected parameters")

        if save_directory is not None:
            # we want to save the instances indexes in a file
            name = self.dataset_name.split(os.sep)[-1].split('.')[0]
            if not os.path.isdir(save_directory):
                os.mkdir(save_directory)
            base_directory = save_directory

            if instances_id is None:
                complete_name = base_directory + os.sep + name + ".instances"
            else:
                complete_name = base_directory + os.sep + name + "." + str(instances_id) + ".instances"
            data = {"dataset": name,
                    "n": len(instances_indexes),
                    "indexes": instances_indexes}

            json_string = json.dumps(data)
            with open(complete_name, 'w') as outfile:
                json.dump(json_string, outfile)

            Tools.verbose("Indexes of selected instances saved in:", complete_name)
        Tools.verbose("Number of instances selected:", len(instances))
        Tools.verbose("----------------------------------------------")
        if details is True:
            if len(instances_details) == 0 and n == 1:
                return None, None
            return instances_details if n is None or n > 1 else instances_details[0] 
        
        if len(instances) == 0 and n == 1:
            return None, None
        return instances if n is None or n > 1 else instances[0]
        

    def predict(self, model, instances: pandas.DataFrame):
        instances = instances.copy()
        # Allow for specific predict functions for the different learners
        if isinstance(instances, pandas.DataFrame) and 'subset' in instances.columns:
            instances = instances.drop(columns='subset')
        instances = self._update_instances(instances, self.get_item_function)
        
        predictions = model.predict(instances)
        if self.labels_type==LabelsType.DiscreteValues:
            predictions = predictions.round()
        return predictions

    
    def _update_instances(self, instances: pandas.DataFrame, get_item_function: Callable) -> pandas.DataFrame|numpy.ndarray:
        instance_reference = get_item_function(instances.iloc[0])
        if isinstance(instance_reference, pandas.Series):
            new_instances = pandas.DataFrame(index=instances.index, columns=instance_reference.index)
            for i, row in instances.iterrows():
                new_instances.loc[i] = get_item_function(row)
            new_instances = new_instances.infer_objects()
        else:
            new_instances = numpy.array([get_item_function(row) for _, row in instances.iterrows()])
        return new_instances
        

    def _get_raw_model(self, model, indexes, is_correct):
        if isinstance(model, Iterable):
            raise ValueError("The model can't be an Iterable.")
        elif model is None:
            assert indexes == InstancesSet.All, "Please insert the model to use this parameter !"
            assert is_correct is None, "Please insert the model to use this parameter !"
            return None, None
        elif isinstance(model, BaseModel):
            return model.learner_information.raw_model, model.learner_information 
        else:
            raise ValueError("The model is not readable: ", str(type(model)))

    
    def _build_original_indexes(self, indexes, seed):
        indexes = indexes.to_list()
        if seed is None: 
            random.shuffle(indexes)
        elif seed != 0:
            random.Random(seed).shuffle(indexes)
        return indexes

    
    def _get_possible_indexes(self, indexes, instances_id, learner_information, train_indexes, test_indexes):
        if train_indexes is not None:
            learner_information.train_index = train_indexes
        elif test_indexes is not None and self.instances_dataframe is not None:
            learner_information.train_index = self.instances_dataframe.index.difference(test_indexes)
        if test_indexes is not None:
            learner_information.test_index = test_indexes
        elif train_indexes is not None and self.instances_dataframe is not None:
            learner_information.train_index = self.instances_dataframe.index.difference(train_indexes)

        if indexes == InstancesSet.Train:
            if learner_information.train_index is None:
                raise ValueError("The training indexes are unknown: please use the train_indexes parameter.")
            possible_indexes = learner_information.train_index
        elif indexes == InstancesSet.Test:
            if learner_information.test_index is None:
                raise ValueError("The testing indexes are unknown: please use the test_indexes parameter.")
            possible_indexes = learner_information.test_index
        elif indexes == InstancesSet.TrainInPriority:
            if learner_information.train_index is None:
                raise ValueError("The training indexes are unknown: please use the train_indexes parameters.")
            if learner_information.test_index is None:
                raise ValueError("The testing indexes are unknown: please use the test_indexes parameter.")
            possible_indexes = numpy.concatenate((learner_information.train_index, learner_information.test_index))
        elif indexes == InstancesSet.All:
            return None 
            
        elif isinstance(indexes, str):
            if os.path.isfile(indexes):
                files_indexes = indexes
            elif os.path.isdir(indexes):
                if instances_id is None:
                    found = False
                    for filename in os.listdir(indexes):
                        file = os.path.join(indexes, filename)
                        if os.path.isfile(file) and file.endswith(".instances"):
                            files_indexes = file
                            assert not found, "Too many .instances files in the directory: " + indexes + ". Please put directly the good file in the option or use the instances_id parameter."
                            found = True
                else:
                    found = False
                    for filename in os.listdir(indexes):
                        file = os.path.join(indexes, filename)
                        if os.path.isfile(file) and file.endswith("." + str(instances_id) + ".instances"):
                            files_indexes = file
                            found = True
                            break
                    assert found, "No ." + str(instances_id) + ".instances" + " files in the directory: " + indexes + " Please put directly the good file in the option or use the instances_id parameter !"
                    Tools.verbose("Loading instances file:", files_indexes)
            f = open(files_indexes)
            data = json.loads(json.load(f))
            possible_indexes = data['indexes']
            f.close()
            
        else:
            raise ValueError("Bad value for 'indexes':", str(indexes))
            
        return possible_indexes

    
    def _get_data(self, dataset: pandas.DataFrame|str|None,
                  possible_indexes: pandas.Index, 
                  subset_true_classes: list|None) -> tuple[pandas.Series, pandas.DataFrame, pandas.Series]:
        if dataset is None:
            if self.instances_dataframe is None:
                raise ValueError("Dataset is not loaded yet: you must put the dataset filename through the 'dataset' parameter.")
        else:
            _, _ = self._parse_data(dataset)
        if possible_indexes is None:
            instances_dataframe, labels_dataframe = self.instances_dataframe.copy(), self.labels_dataframe.copy()
        else:
            instances_dataframe, labels_dataframe = self.instances_dataframe.copy().loc[possible_indexes], self.labels_dataframe.copy().loc[possible_indexes]
        if subset_true_classes is not None:
            instances_dataframe = instances_dataframe[[x in subset_true_classes for x in labels_dataframe.label]]
            labels_dataframe = labels_dataframe[[x in subset_true_classes for x in labels_dataframe.label]]
        instances_dataframe = instances_dataframe.infer_objects()
        return instances_dataframe.index, instances_dataframe, labels_dataframe


    def _create_learner_information(self, model, train_index=None, test_index=None, group=None, metrics=None, extras={}):
        return LearnerInformation(
                    self.get_learner_name(), 
                    self.problem_type, self.model_type, self.instances_type, self.labels_type, self.splitting_method,
                    self.dataset_path, self.n_features, self.n_labels, self.feature_names, self.classes, self.instances_directory, self.labels_directory, self.get_item_function,
                    model, metrics, extras.copy(), train_index, test_index, group)
    
    def _synchronize_learner_information(self, learner_information):
        self.problem_type = learner_information.problem_type
        self.model_type = learner_information.model_type
        self.instances_type = learner_information.instances_type
        self.labels_type = learner_information.labels_type
        self.splitting_method = learner_information.splitting_method

        self.dataset_path = learner_information.dataset_path
        self.n_features = learner_information.n_features
        self.n_labels = learner_information.n_labels
        self.feature_names = learner_information.feature_names
        self.classes = learner_information.label_names
        self.instances_directory = learner_information.instances_directory
        self.labels_directory = learner_information.labels_directory
        self.get_item_function = learner_information.get_item_function
        

        