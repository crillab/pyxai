import os
import json
import pickle
import pandas 

from typing import Callable, Iterable
from shutil import rmtree

from pyxai.sources.learners.learner import Learner, NoneData, LearnerInformation
from pyxai.sources.learners.xgboost import Xgboost
from pyxai.sources.learners.scikitlearn import Scikitlearn
from pyxai.sources.learners.lightgbm import LightGBM

from pyxai.sources.type import ProblemType, ModelType, InstancesType, LabelsType

from pyxai import Tools



class ModelIO():
    r"""A utility class designed to handle input/output operations for machine learning models of PyXAI. 
    This class covers importing, loading, and saving models.
    """

    def import_models(models,
                  data: str|pandas.DataFrame|NoneData = NoneData, *,
                  problem_type: str|ProblemType|None = None,
                  model_type: str|ModelType|None = None,
                  instances_type: str|InstancesType|None = None,
                  labels_type: str|LabelsType|None = None,
                  get_item_function: Callable|None = None,
                  instances_directory: str|None = None,
                  labels_directory: str|None = None):
        
        r"""Import existing ML models. 
            
            The method detects the type of models and applies the correct conversions in order to translate them into PyXAI data structures. 
            
            Parameters
            ----------

            models: ```list[RandomForestClassifier|DecisionTreeClassifier|XGBClassifier|XGBRegressor|LGBMRegressor]```
                List of models to import.
                Possible values are defined in the ```SplittingMethod``` enum.

            data: ```str``` | ```pandas.DataFrame``` | ```NoneData``` (optional, default=```NoneData```)
                The dataset to use, either as a path to a csv, json or excel file or as a pandas DataFrame.

            model_type: ```str``` | ```ModelType``` 
                The type of model (linear, tree-based, neural network, ...)
                Possible values are defined in the ```ModelType``` enum.
            
            instances_type: ```str``` | ```InstancesType``` (optional, default=```None```)
                The type of instances (image, tabular, text, temporal, ...)
                Possible values are defined in the ```InstancesType``` enum.
                    
            labels_type: ```str``` | ```LabelsType``` (optional, default=```None```)
                The type of labels (class, text, mask, contours, ...)
                Possible values are defined in the ```LabelsType``` enum.

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

            Returns
            -------
            ```tuple(Scikitlearn|Xgboost|LightGBM, list[DecisionTree|RandomForest|BoostedTrees|BoostedTreesRegression])```:
                Return a tuple (learner, models) with the good learner and the PyXAI models. 

            Examples
            --------
            >>> from pyxai import Tools, Learning, Explaining
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn import datasets
            
            >>> model_rf = RandomForestClassifier(random_state=0)
            >>> data = datasets.load_breast_cancer(as_frame=True)
            >>> X = data.data
            >>> Y = data.target

            >>> feature_names = data.feature_names
            >>> model_rf.fit(X, Y)

            >>> learner, model = Learning.ModelIO.import_models(model_rf, instances_type='tabular')
            >>> learner.feature_names = feature_names
            >>> instance, prediction = learner.get_instances(dataset=data.frame, model=model, n=1)
            
            >>> explainer = Explaining.initialize(model, instance=instance)
            >>> direct = explainer.direct_reason()
                        
        """
        
        if not isinstance(models, (list, tuple)):
            models = [models]
        if not all(type(model) == type(models[0]) for model in models):
            raise ValueError("All models must be of the same type: " + str(type(models[0])))

        if type(models[0]) in Scikitlearn.get_problem_types().keys():
            problem_type = Scikitlearn.get_problem_types()[type(models[0])][0]
            model_type = Scikitlearn.get_problem_types()[type(models[0])][1]
            learner = Scikitlearn(data, problem_type=problem_type, model_type=model_type, instances_type=instances_type, labels_type=labels_type,
                                get_item_function=get_item_function, instances_directory=instances_directory, labels_directory=labels_directory)
            extras = {
                "learner": str(type(models[0])),
                "learner_options": models[0].get_params(),
                "base_score": 0,
            }
            
        elif type(models[0]) in Xgboost.get_problem_types().keys():
            problem_type = Xgboost.get_problem_types()[type(models[0])][0]
            model_type = Xgboost.get_problem_types()[type(models[0])][1]
            learner = Xgboost(data, problem_type=problem_type, model_type=model_type, instances_type=instances_type, labels_type=labels_type,
                            get_item_function=get_item_function, instances_directory=instances_directory, labels_directory=labels_directory)
            extras = {
                "learner": str(type(models[0])),
                "learner_options": models[0].get_xgb_params(),
                "base_score": None, # It is set in learner.convert_model
            }

        elif type(models[0]) in LightGBM.get_problem_types().keys():
            problem_type = LightGBM.get_problem_types()[type(models[0])][0]
            model_type = LightGBM.get_problem_types()[type(models[0])][1]
            learner = LightGBM(data, problem_type=problem_type, model_type=model_type, instances_type=instances_type, labels_type=labels_type,
                            get_item_function=get_item_function, instances_directory=instances_directory, labels_directory=labels_directory)
            extras = {
                "learner": str(type(learner)),
                "learner_options": models[0].get_params(),
                "base_score": 0
            }
            
        else:
            raise ValueError("The type of this model is unknown: "+str(type(models[0])))

        learner_information_lst=[learner._create_learner_information(model, extras=extras) for model in models]
        result_output = learner.convert_model(learner_information_lst)
        Tools.verbose("---------------   Explainer   ----------------")
        for i, result in enumerate(result_output):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose(result)
        return learner, result_output if len(result_output) != 1 else result_output[0]


    def save(models, save_directory):
        r"""Save models in the ```save_directory``` in the form of two files: ```<save_directory>/<dataset>.<i>.map``` and ```<save_directory>/<dataset>.<i>.pkl``` where ```<i>``` is the index of a model given in the model parameter. 
            The backup formats of .pkl of models are Pickle files. The .map files contain information about models in JSON formats.   
            
            Parameters
            ----------

            models: ```list[DecisionTree|RandomForest|BoostedTrees|BoostedTreesRegression] | DecisionTree|RandomForest|BoostedTrees|BoostedTreesRegression```
                PyXAI model or List of PyXAI models to be saved.

            save_directory: ```str```
                The directory where the models are saved. Creates the directory if it does not exist.

            Returns
            -------
            ```tuple(Scikitlearn|Xgboost|LightGBM, list[DecisionTree|RandomForest|BoostedTrees|BoostedTreesRegression])```:
                Return a tuple (learner, models) with the good learner and the PyXAI models. 

            Examples
            --------
            >>> from pyxai import Tools, Learning
            
            >>> learner = Learning.Scikitlearn(dataset, problem_type=problem_type, labels_type=labels_type)
            >>> model = learner.evaluate(splitting_method='hold-out', model_type=model_type, splitting_parameters={"test_size":0.2, "random_state":0})
        
            >>> Learning.ModelIO.save(model, "saving_directory")
                        
        """
        if not isinstance(models, Iterable): models = [models]
        name = models[0].learner_information.dataset_path.split(os.sep)[-1].split('.')[0]
        if save_directory is not None:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            base_directory = save_directory
        else:
            base_directory = ""

        rmtree(base_directory, ignore_errors=True)
        os.mkdir(base_directory)

        for i, model in enumerate(models):
            learner_information = model.learner_information
            filename = base_directory + os.sep + name + '_' + str(i)
            # save the model:
            model_filename = ModelIO._save_model(learner_information.raw_model, filename)
            
            # save some informations
            learner_information_file_name = ModelIO._save_learner_information(learner_information, filename)

            Tools.verbose("Model saved: ("+ model_filename + ", " + learner_information_file_name)

    def load(models_directory, *, data=NoneData, tests=False): 
        r"""Load the models that were saved using the ```save``` method from the ```models_directory``` directory. 
            The files in the ```models_directory``` have to be in the form of two files per model:  
                - ```<models_directory>/<dataset>.<i>.pkl```: the models in the form of Pickle files  
                - ```<models_directory>/<dataset>.<i>.map```: contain information about models in JSON formats.  
            where ```<i>``` is the index of a model.
            
            Parameters
            ----------

            models_directory: ```str```
                The directory of models to load.

            data: ```str``` | ```pandas.DataFrame``` | ```NoneData``` (optional, default=```NoneData```)
                The dataset to use, either as a path to a csv, json or excel file or as a pandas DataFrame.                
            
            Examples
            --------
            >>> from pyxai import Tools, Learning
            
            >>> learner = Learning.Scikitlearn(dataset, problem_type=problem_type, labels_type=labels_type)
            >>> model = learner.evaluate(splitting_method='hold-out', model_type=model_type, splitting_parameters={"test_size":0.2, "random_state":0})
        
            >>> Learning.ModelIO.save(model, "my_models")
            
            >>> del learner
            >>> del model
            
            >>> learner, model = Learning.ModelIO.load("my_models", dataset=dataset)
            >>> instances = learner.get_instances(model=model, n=n_instances, indexes=Learning.TEST)     
        """
        files = ModelIO._load_get_files(models_directory)

        # Initialize all learner informations 
        learner_informations = []
        for pkl_file, map_file in files: 
            learner_informations.append(
                LearnerInformation.__init__from__json__(ModelIO._load_learner_information(map_file), ModelIO._load_model(pkl_file)))

        # Check that some parameters of learner informations are the same
        learner_parameters = ['learner_name', 'problem_type', 'model_type',
                            'instances_type', 'labels_type', 'get_item_function',
                            'instances_directory', 'labels_directory']

        for parameter in learner_parameters:
            if parameter != "get_item_function": #One loaded, these functions are differents
                if not all(learner_information.to_dict()[parameter] == learner_informations[0].to_dict()[parameter] for learner_information in learner_informations):
                    raise ValueError(f"All learners must have the same {parameter}: {learner_informations[0].to_dict()[parameter]}.")

        # Build the learner
        dict_learner_parameters = {key: learner_informations[0].to_dict()[key] for key in learner_parameters if key != 'learner_name'}
        learner_name = learner_informations[0].to_dict()["learner_name"]
        if learner_name == Xgboost.__name__:
            learner = Xgboost(data, **dict_learner_parameters)
        elif learner_name == Scikitlearn.__name__:
            learner = Scikitlearn(data, **dict_learner_parameters)
        elif learner_name == LightGBM.__name__:
            learner = LightGBM(data, **dict_learner_parameters)
        else:
            raise ValueError("The learner name is unknown:" + str(learner_name))
        
        # Add learner_informations to the learner
        learner.learner_information_lst = learner_informations
        learner._synchronize_learner_information(learner_informations[0])
        
        Tools.verbose("---------   Evaluation Information   ---------")
        for i, learner_information in enumerate(learner.learner_information_lst):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose("mapping file:", files[i][1])
            Tools.verbose("nFeatures (nAttributes, with the labels):", learner.n_features)
            Tools.verbose("nInstances (nObservations):", len(learner_information.train_index) + len(learner_information.test_index))
            Tools.verbose("nLabels:", learner.n_labels)
            Tools.verbose("metrics:", learner_information.metrics)
            Tools.verbose("Number of Training instances:", len(learner_information.train_index))
            Tools.verbose("Number of Testing instances:", len(learner_information.test_index))
            Tools.verbose()

            # Test phase
            if tests:
                instances_test = learner.instances_dataframe.iloc[learner_information.test_index]
                labels_test = learner.labels_dataframe.label.iloc[learner_information.test_index]
                result = learner.predict(learner_information.raw_model, instances_test)
                metrics = learner.compute_metrics(labels_test, result)
                if metrics != learner_information.metrics:
                    raise ValueError("The calculated metrics are no longer the same as in the backup file.")
                
        #Convert the models into PyXAI models
        pyxai_models = learner.convert_model()
        
        Tools.verbose("---------------   Explainer   ----------------")
        for i, result in enumerate(pyxai_models):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose(result)

        return learner, pyxai_models if len(pyxai_models) != 1 else pyxai_models[0]
    
    def _load_get_files(models_directory):
        assert models_directory is not None and os.path.exists(models_directory), "The path of models_directory do not exist: " + str(models_directory)    
        files = []
        index = 0
        found = True
        while found:
            found = False
            for filename in os.listdir(models_directory):
                model_file = os.path.join(models_directory, filename)
                if os.path.isfile(model_file) and model_file.endswith(str(index) + ".pkl"):
                    map_file = model_file.replace(".pkl", ".map")
                    assert os.path.isfile(map_file), "A '.pkl' file must be accompanied by a '.map' file !"
                    files.append((model_file, map_file))
                    index += 1
                    found = True
                    break
                
        assert len(files) != 0, "No file representing a model in the path: " + models_directory
        return files

    def _save_model(raw_model, filename):
        file = open(filename + ".pkl", 'wb')
        pickle.dump(raw_model, file)
        file.close()
        return filename + ".pkl"
    
    def _load_model(file_path):
        model = None
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    def _save_learner_information(learner_information, filename):
        json_string = json.dumps(learner_information.to_json_dict())
        with open(filename + ".map", 'w') as outfile:
            json.dump(json_string, outfile)
        return filename + ".map"
    
    def _load_learner_information(file_path):
        learner_information = None
        with open(file_path) as f:
            learner_information = json.loads(json.load(f))
        return learner_information
