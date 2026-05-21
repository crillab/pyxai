from pandas import Series as pd_Series
from numpy import array as np_array
from numpy import ndarray as np_ndarray

from pyxai.sources.learners.learner_information import LearnerInformation
from pyxai.sources.type import ProblemType, InstancesType, LabelsType

class BaseModel():
    def __init__(self, n_features: int|None = None,
                 n_classes: int|None = None,
                 learner_information: LearnerInformation|None = None,
                 feature_names: list[int|str]|None = None,
                 class_names: list[int|str]|None = None):
        self.n_features: int|None = n_features # Number of features
        self.learner_information = learner_information
        self.feature_names: list[int|str]|None = feature_names
        self.n_classes: int|None = n_classes
        self.class_names: list[int|str]|None = class_names
        if self.learner_information is not None:
            self._adjust_features()
            self._adjust_classes()
        

    def _adapt_to_problem(self, value: int|float) -> str|int|float: 
        if self.learner_information.problem_type == ProblemType.Classification and self.class_names is not None: 
            return self.class_names[value]
        elif self.learner_information.problem_type == ProblemType.Regression:
            if isinstance(value, np_ndarray) and value.ndim==1 and value.shape[0]==1:
                value = value[0]
            if self.learner_information.labels_type == LabelsType.DiscreteValues:
                return round(value)
        return value
        
    def predict_instance(self, instance: pd_Series|np_ndarray) -> str|int|float:
        if isinstance(instance, pd_Series) and 'subset' in instance.index:
            instance = instance.drop(index='subset')
    
        instance = self.learner_information.get_item_function(instance)
        return self._adapt_to_problem(self._predict_instance(instance))
        
    def predict_implicant(self, implicant: np_ndarray) -> str|int|float:
        return self._adapt_to_problem(self._predict_implicant(implicant))
    
    def _adjust_classes(self):  
        if self.learner_information.problem_type == ProblemType.Classification:
            if self.class_names is None and self.learner_information.label_names is not None:
                self.class_names = self.learner_information.label_names
            elif self.class_names is not None and self.learner_information.label_names is None:
                self.learner_information.label_names = self.class_names
            else:
                pass # TODO Add Error if they already exists but are different (needs to check element by element)

            if self.class_names is not None:
                if self.n_classes is None:
                    self.n_classes = len(self.class_names)
                elif self.n_classes != len(self.class_names):
                    raise ValueError("The number of classes doesn't correspond to the number of class names")                
                
    def _adjust_features(self):
        if self.feature_names is None and self.learner_information.feature_names is not None:
            self.feature_names = self.learner_information.feature_names
        elif self.feature_names is not None and self.learner_information.feature_names is None:
            self.learner_information.feature_names = self.feature_names
        else:
            pass # TODO Add Error if they already exists but are different (needs to check element by element)
            
        if self.feature_names is not None:
            if self.n_features is None:
                self.n_features = len(self.feature_names)
            elif self.n_features != len(self.feature_names):
                raise ValueError("When the `feature_names` parameter is used, it must be a list with `n_features` names.")
            

    @property
    def raw_model(self):
        return self.learner_information.raw_model


    @property
    def accuracy(self):
        return self.learner_information.accuracy


    @property
    def features_name(self):
        return self.learner_information.feature_names