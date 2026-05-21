

import dill
import base64

from pyxai.sources.type import ProblemType, ModelType, InstancesType, LabelsType, SplittingMethod

class LearnerInformation:
    r"""This class contains some information about a learner, a dataset and a model.
    See the ```__init__``` method for more precision.
    """

    def __init__(self, 
        # About the learner
                 learner_name=None, 
                 problem_type=None, 
                 model_type=None,
                 instances_type=None, 
                 labels_type=None, 
                 splitting_method=None, 
        # About the dataset
                 dataset_path=None,
                 n_features=None,
                 n_labels=None,
                 feature_names=None, 
                 label_names=None, 
                 instances_directory=None,
                 labels_directory=None,
                 get_item_function=None,
        # About the model
                 raw_model=None, 
                 metrics=None, 
                 extras={},
                 train_index=None, 
                 test_index=None, 
                 groups=None, 
                 ):
        r"""
        Set several information about a model, a dataset and a model.

        Parameters
        ----------
        learner_name: ```str``` (optional, default=None)
            The name of the library used for the evaluation. 

        problem_type: ```str``` | ```ProblemType``` (optional, default=None)
            The type of problem (classification, regression, ...)
            Possible values are defined in the ```ProblemType``` enum.

        model_type: ```str``` | ```ModelType``` (optional, default=None)
            The type of model (linear, tree-based, neural network, ...)
            Possible values are defined in the ```ModelType``` enum.

        instances_type: ```str``` | ```InstancesType``` (optional, default=None)
            The type of instances (image, tabular, text, temporal, ...)
            Possible values are defined in the ```InstancesType``` enum.
                
        labels_type: ```str``` | ```LabelsType``` (optional, default=None)
            The type of labels (class, text, mask, contours, ...)
            Possible values are defined in the ```LabelsType``` enum.

        splitting_method: ```str``` | ```SplittingMethod``` (optional, default=None)
            The splitting method used for the evalution (hold-out, k-folds, ...)
            Possible values are defined in the ```SplittingMethod``` enum.


        dataset_path: ```str``` (optional, default=None)
            The path and the filename of the dataset.

        n_features: ```int``` (optional, default=None)
            The number of features. 
        
        n_labels: ```int``` (optional, default=None)
            The number of labels. 

        feature_names : ```list``` of ```str``` (optional, default=None)
            The names of the features used in the model.  

        label_names: ```list``` of ```str``` (optional, default=None)
            The name of the labels (without redundancy)

        instances_directory: ```str``` (optional, default=None)
            The directory of instances. 

        labels_directory: ```str``` (optional, default=None)
            The directory of labels. 

        get_item_function: ```Callable``` (optional, default=None)
            A function to get an instance from the dataset. This function is used to get an instance in the right format for the model.
            If the dataset is a pandas DataFrame and the instances are tabular, this function is not necessary and can be set to None. 
            In other cases, this function should be defined by the user. It should take as input a row of the dataframe and return the corresponding instance in the right format for the model.

        
        raw_model: ```DecisionTreeClassifier``` | ```RandomForestClassifier``` | ```XGBClassifier``` | ```XGBRegressor``` | ```LGBMRegressor``` (optional, default=None)
            The raw model of the librairy used to create the model (sklearn, xgboost, lightgmb, ...)

        metrics: ```dict``` (optional, default=None)
            A dictionnary containing some metrics about the evaluation (precision, recall, f1_score, specificity, ...)
            More information are given in the ```Metrics``` page.

        extras: ```dict``` (optional, default={})
            Extra information from the raw model (the type, model parameters, base_score, ...)
    
        train_index: ```list``` of ```int``` (optional, default=None)
            A list of training indexes used during the evaluation (cross-validation)

        test_index: ```list``` of ```int``` (optional, default=None)
            A list of test indexes used during the evaluation (cross-validation)

        groups: ```list``` of ```int``` (optional, default=None)
            The used groups for the leave-one-group-out evaluation
        
        """

        #About the learner
        self.learner_name = learner_name
        self.problem_type = problem_type
        self.model_type = model_type
        self.instances_type = instances_type
        self.labels_type = labels_type
        self.splitting_method = splitting_method
        
        #About the data
        self.dataset_path = dataset_path
        self.n_features = n_features
        self.n_labels = n_labels
        self.feature_names = feature_names
        self.label_names = label_names # <=> class_names
        self.instances_directory = instances_directory
        self.labels_directory = labels_directory
        self.get_item_function = get_item_function
        
        #About the model
        self.raw_model = raw_model
        self.metrics = metrics
        self.extras = extras
        
        self.train_index = train_index
        self.test_index = test_index
        self.groups = groups

    @classmethod
    def __init__from__json__(cls, json_dict, raw_model):
        return LearnerInformation(  json_dict["learner_name"], 
                                    ProblemType(json_dict["problem_type"]),
                                    ModelType(json_dict["model_type"]), 
                                    InstancesType(json_dict["instances_type"]),
                                    LabelsType(json_dict["labels_type"]),
                                    SplittingMethod(json_dict["splitting_method"]),
                                    json_dict["dataset_path"],
                                    json_dict["n_features"],
                                    json_dict["n_labels"], 
                                    json_dict["feature_names"],
                                    json_dict["label_names"],
                                    json_dict["instances_directory"],
                                    json_dict["labels_directory"],
                                    dill.loads(base64.b64decode(json_dict["get_item_function"].encode('utf-8'))) if json_dict["get_item_function"] is not None else None,
                                    raw_model,
                                    json_dict["metrics"],
                                    json_dict["extras"],
                                    json_dict["train_index"],
                                    json_dict["test_index"],
                                    json_dict["groups"] 
                                  ) 

    def to_json_dict(self):
        json_dict = self.__dict__.copy()
        
        del json_dict["raw_model"]

        json_dict["problem_type"] = self.problem_type if type(self.problem_type)==str else self.problem_type.value
        json_dict["model_type"] = self.model_type if type(self.model_type)==str else self.model_type.value
        json_dict["instances_type"] = self.instances_type if type(self.instances_type)==str else self.instances_type.value
        json_dict["labels_type"] = self.labels_type if type(self.labels_type)==str else self.labels_type.value
        json_dict["splitting_method"] = self.splitting_method if type(self.splitting_method)==str else self.splitting_method.value
        
        json_dict["label_names"] = None if self.label_names is None else self.label_names.tolist()
        json_dict["train_index"] = None if self.train_index is None else self.train_index.tolist()
        json_dict["test_index"] = None if self.test_index is None else self.test_index.tolist()
        
        json_dict["get_item_function"] = base64.b64encode(dill.dumps(self.get_item_function)).decode('utf-8')

        return json_dict
    
    def to_dict(self):
        return self.__dict__
            

