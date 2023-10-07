import copy
import json
import os
import random
import shutil
from collections import OrderedDict
from typing import Iterable

import numpy
import pandas
from sklearn.model_selection import LeaveOneGroupOut, train_test_split, KFold

from pyxai import Tools
from pyxai.sources.core.structure.boostedTrees import BoostedTrees
from pyxai.sources.core.structure.decisionTree import DecisionTree
from pyxai.sources.core.structure.randomForest import RandomForest
from pyxai.sources.core.structure.type import EvaluationMethod, LearnerType, EvaluationOutput, Indexes, TypeFeature
from pyxai.sources.core.tools.utils import flatten, compute_accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyxai.sources.learning.learner_information import LearnerInformation
        
class NoneData: pass

class Learner:
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    """
    def __init__(self, data=NoneData, learner_type=None):
        if learner_type is None:
            raise ValueError("Please set the parameter 'learner_type' to 'Learning.CLASSIFICATION' or 'Learning.REGRESSION'.")
        self.learner_type = learner_type
        self.dict_labels = None
        self.data = None
        self.labels = None
        self.n_labels = None
        self.n_features = None
        self.dataset_name = None
        self.n_instances = None
        self.feature_names = None
        self.n_categorical = None
        self.n_numerical = None
        self.learner_information = []
        
        data, name = self.parse_data(data)
        if data is not None: 
            self.load_data(data, name)
    
    def get_details(self):
        return [learner_information.to_dict() for learner_information in self.learner_information]

    def get_raw_models(self):
        return [learner_information.raw_model for learner_information in self.learner_information]
    
    def parse_data(self, data=NoneData, skip=None, n=None):
        if data is NoneData:
            return None, None

        if isinstance(data, str):
            if os.path.isfile(data):
                if data.endswith(".csv"):
                    return pandas.read_csv(data, skiprows=skip, nrows=n).copy(), data
                elif data.endswith(".xlsx"):
                    return pandas.read_excel(data, skiprows=skip, nrows=n).copy(), data
                else:
                    raise ValueError("If data is a string, it must be a csv or excel file with the .csv or .xlsx extension.")
            raise FileNotFoundError("The file " + data + " is not found.")

        if isinstance(data, pandas.core.frame.DataFrame):
            return data, "pandas.core.frame.DataFrame"

        if data is None:
            raise ValueError("The 'data' parameter is set but None (can be optional, but not None). If you are passed 'Tools.Options.dataset' as parameter, check if the '-dataset=<dataset_file>' option is present in your command.")

        raise ValueError("The data parameter is either a string representing a csv or excel file or a pandas.core.frame.DataFrame object.")


    @staticmethod
    def count_lines(filename):
        with open(filename) as f:
            return sum(1 for _ in f)


    def load_data_limited(self, datasetname, possible_indexes, n):
        self.dataset_name = datasetname
        n_indexes = self.count_lines(datasetname) - 1  # to skip the first line
        if possible_indexes is None:
            skip = None
        else:
            # To avoid a 'i not in possibles_indexes' in a loop.
            map_possible_indexes = [False for _ in range(n_indexes)]
            for v in possible_indexes: map_possible_indexes[v] = True 
            # Create the values to skip
            skip = [i+1 for i in range(n_indexes) if map_possible_indexes[i] is False]
            # Create a map to get the good order of instances
            sorted_possibles_indexes = sorted(possible_indexes)
            map_possibles_indexes = [sorted_possibles_indexes.index(index) for index in possible_indexes]

        data = pandas.read_csv(
            datasetname,
            skiprows=skip,
            nrows=n
        )
        # recreate the dataframe object but with the good order of instances
        if skip is not None:
            sorted_data = pandas.DataFrame(columns=data.columns).astype(data.dtypes)
            for i in range(data.shape[0]):
                #sorted_data = sorted_data.append(data.loc[map_possibles_indexes[i]].to_dict(), ignore_index=True)
                pandas.concat([sorted_data, data.loc[map_possibles_indexes[i]]], ignore_index=True)
            sorted_data = sorted_data.astype(data.dtypes)

        n_instances, n_features = data.shape
        self.feature_names = data.columns.values.tolist()
        self.rename_attributes(data)
        data, labels = self.remove_labels(data, n_features)
        if self.dict_labels is not None:
            labels = self.labels_to_values(labels)
        data = data.to_numpy()
        return data, labels


    def load_data(self, dataframe, datasetname, *, encode_labels=True):
        """
        dataframe: A pandas.core.frame.DataFrame object.
        """
        self.dataset_name = datasetname
        self.data = dataframe
        Tools.verbose("data:")
        Tools.verbose(self.data)
        self.n_instances, self.n_features = self.data.shape
        self.feature_names = self.data.columns.values.tolist()

        self.rename_attributes(self.data)

        self.data, self.labels = self.remove_labels(self.data, self.n_features)
        if self.learner_type == LearnerType.Classification and encode_labels is True:
            self.create_dict_labels(self.labels)
            self.labels = self.labels_to_values(self.labels)
            self.n_labels = len(set(self.labels))
            
        self.data = self.data.to_numpy()  # remove the first line (attributes) and now the first dimension represents the instances :)!
        Tools.verbose("--------------   Information   ---------------")
        Tools.verbose("Dataset name:", self.dataset_name)
        Tools.verbose("nFeatures (nAttributes, with the labels):", self.n_features)
        Tools.verbose("nInstances (nObservations):", self.n_instances)
        Tools.verbose("nLabels:", self.n_labels)
        if self.n_labels == 1:
            raise ValueError("The prediction contains only one value: " + str(self.n_labels))

    """
    Rename attributes in self.data in string of integers from 0 to 'self.n_attributes'
    """


    @staticmethod
    def rename_attributes(data):
        rename_dictionary = {element: str(i) for i, element in enumerate(data.columns)}
        data.rename(columns=rename_dictionary, inplace=True)

    def get_label_from_value(self, value):
        return self.inverse_dict_labels[value]

    def create_dict_labels(self, labels):
        index = 0
        self.dict_labels = OrderedDict()
        self.inverse_dict_labels = OrderedDict()
        for p in labels:
            if str(p) not in self.dict_labels:
                self.dict_labels[str(p)] = index
                self.inverse_dict_labels[index] = str(p)
                index += 1


    """
    Convert labels (predictions) into binary values
    Using of OrderedDict in order to be reproducible.
    """


    def labels_to_values(self, labels):
        return [self.dict_labels[str(element)] for element in labels]


    """
    Remove and get the prediction: it is the last attribute (column) in the file
    """


    @staticmethod
    def remove_labels(data, n_features):
        prediction = data[str(n_features - 1)].copy().to_numpy()
        data = data.drop(columns=[str(n_features - 1)])
        return data, prediction

    def compute_metrics(self, true, prediction):
        if self.learner_type == LearnerType.Classification:
            return {"accuracy": compute_accuracy(prediction, true)}
        elif self.learner_type == LearnerType.Regression:
            return {
                "mean_squared_error": mean_squared_error(true, prediction),
                "root_mean_squared_error": mean_squared_error(true, prediction, squared=False),
                "mean_absolute_error": mean_absolute_error(true, prediction)
            }
        else:
            raise ValueError("learner_type is unknown:", str(self.learner_type))

    def evaluate(self, *, method, output, n_models=10, test_size=0.3, **learner_options):
        if "seed" not in learner_options.keys():
            learner_options["seed"] = 0
        if "max_depth" not in learner_options.keys():
            learner_options["max_depth"] = None
        
        
        Tools.verbose("---------------   Evaluation   ---------------")
        Tools.verbose("method:", str(method))
        Tools.verbose("output:", str(output))
        Tools.verbose("learner_type:", str(self.learner_type))

        if method == EvaluationMethod.HoldOut:
            self.hold_out_evaluation(output, test_size=test_size, learner_options=learner_options)
        elif method == EvaluationMethod.LeaveOneGroupOut:
            self.leave_one_group_out_evaluation(output, n_trees=n_models, learner_options=learner_options)
        elif method == EvaluationMethod.KFolds:
            self.k_folds_evaluation(output, n_models=n_models, learner_options=learner_options)
        else:
            assert False, "Not implemented !"

        for learner_information in self.learner_information:
            learner_information.set_learner_name(self.get_learner_name())
            learner_information.set_feature_names(self.feature_names)
            learner_information.set_evaluation_method(method)
            learner_information.set_evaluation_output(output)

        Tools.verbose("---------   Evaluation Information   ---------")
        for i, result in enumerate(self.learner_information):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose("metrics:")
            for key in result.metrics.keys():
                Tools.verbose("   " + key + ": " +str(result.metrics[key]))
                
            Tools.verbose("nTraining instances:", len(result.training_index))
            Tools.verbose("nTest instances:", len(result.test_index))
            Tools.verbose()

        Tools.verbose("---------------   Explainer   ----------------")
        result_output = self.convert_model(output)
        # elif output == EvaluationOutput.SAVE:
        #  self.save_model(model_directory)
        #  result_output = self.to_BT()

        for i, result in enumerate(result_output):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose(result)

        # Add the type of features
        
        return result_output if len(result_output) != 1 else result_output[0]

    def convert_model(self, output, learner_information=None):
        if learner_information is not None: 
            self.learner_information = learner_information

        if self.learner_type == LearnerType.Classification:
            if output == EvaluationOutput.DT:
                return self.to_DT_CLS(self.learner_information)
            elif output == EvaluationOutput.RF:
                return self.to_RF_CLS(self.learner_information)
            elif output == EvaluationOutput.BT:
                return self.to_BT_CLS(self.learner_information)
            else:
                raise NotImplementedError(str(output) + " not implemented.")
        elif self.learner_type == LearnerType.Regression:
            if output == EvaluationOutput.DT:
                return self.to_DT_REG(self.learner_information)
            elif output == EvaluationOutput.RF:
                return self.to_RF_REG(self.learner_information)
            elif output == EvaluationOutput.BT:
                return self.to_BT_REG(self.learner_information)
            else:
                raise NotImplementedError(str(output) + " not implemented.")
        else:
            raise NotImplementedError(str(self.learner_type) + " not implemented.")

    def load_get_files(self, models_directory):
        assert models_directory is not None and os.path.exists(models_directory), "The path of models_directory do not exist: " + str(
            models_directory)

        self.learner_information.clear()
        # get the files
        files = []
        index = 0
        found = True
        while found:
            found = False
            for filename in os.listdir(models_directory):
                model_file = os.path.join(models_directory, filename)
                if os.path.isfile(model_file) and model_file.endswith(str(index) + ".model"):
                    map_file = model_file.replace(".model", ".map")
                    assert os.path.isfile(map_file), "A '.model' file must be accompanied by a '.map' file !"
                    files.append((model_file, map_file))
                    index += 1
                    found = True
                    break

        assert len(files) != 0, "No file representing a model in the path: " + models_directory
        return files


    def load(self, *, models_directory, tests=False):
        files = self.load_get_files(models_directory)

        for _, model in enumerate(files):
            model_file, map_file = model

            # recuperate map
            f = open(map_file)
            
            data = json.loads(json.load(f))
            raw_model = self.load_model(model_file, data['extras']['learner_options'])
            learner_information = LearnerInformation(
                                    copy.deepcopy(raw_model), 
                                    data['training_index'], 
                                    data['test_index'], 
                                    None, 
                                    data['metrics'],
                                    data['extras'])
            learner_information.set_learner_name(data['learner_name'])
            learner_information.set_evaluation_method(data['evaluation_method'])
            learner_information.set_evaluation_output(data['evaluation_output'])
            learner_information.set_feature_names(data["feature_names"])
            self.n_features = data['n_features']
            self.n_labels = data["n_labels"]
            self.dict_labels = data["dict_labels"]
            self.feature_names = data["feature_names"]
            f.close()

            if self.get_learner_name() != learner_information.learner_name:
                raise ValueError("The learner in the .map file is not the same: " + self.get_learner_name() + " != " + learner_information.learner_name)
            
            # load model
            Tools.verbose("----------   Loading Information   -----------")
            Tools.verbose("mapping file:", map_file)
            Tools.verbose("nFeatures (nAttributes, with the labels):", self.n_features)
            Tools.verbose("nInstances (nObservations):", len(learner_information.training_index) + len(learner_information.test_index))
            Tools.verbose("nLabels:", self.n_labels)
            if tests:
                # Test phase
                instances_test = [self.data[i] for i in learner_information.test_index]
                labels_test = [self.labels[i] for i in learner_information.test_index]
                result = raw_model.predict(instances_test)
                metrics = self.compute_metrics(labels_test, result)
                if metrics != learner_information.metrics:
                    raise ValueError("The calculated metrics are no longer the same as in the backup file." + metrics + " != " + learner_information.metrics)
            self.learner_information.append(learner_information)
            
        Tools.verbose("---------   Evaluation Information   ---------")
        for i, learner_information in enumerate(self.learner_information):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose("metrics:", learner_information.metrics)
            Tools.verbose("nTraining instances:", len(learner_information.training_index))
            Tools.verbose("nTest instances:", len(learner_information.test_index))
            Tools.verbose()

        Tools.verbose("---------------   Explainer   ----------------")
        output = EvaluationOutput.from_str(self.learner_information[-1].evaluation_output)
        result_output = self.convert_model(output)


        for i, result in enumerate(result_output):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose(result)
        return result_output if len(result_output) != 1 else result_output[0]


    def save(self, models, save_directory, generic=False):
        if not isinstance(models, Iterable): models = [models]

        name = self.dataset_name.split(os.sep)[-1].split('.')[0]
        if save_directory is not None:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            base_directory = save_directory
        else:
            base_directory = ""

        shutil.rmtree(base_directory, ignore_errors=True)
        os.mkdir(base_directory)

        

        for i, trees in enumerate(models):
            learner_information = trees.learner_information
            filename = base_directory + os.sep + name + '.' + str(i)
            # model:
            if not generic:
                self.save_model(learner_information, filename)
            else:
                self.save_model_generic(trees, filename)
            # map of indexes for training and test part
            data = {"learner_name": learner_information.learner_name if not generic else "Generic",
                    "learner_type": str(self.learner_type),
                    "extras": learner_information.extras,
                    "metrics": learner_information.metrics,
                    "evaluation_method": learner_information.evaluation_method,
                    "evaluation_output": learner_information.evaluation_output,
                    "n_features": self.n_features,
                    "n_labels": self.n_labels,
                    "dict_labels": self.dict_labels,
                    "feature_names": self.feature_names,
                    "training_index": learner_information.training_index.tolist(),
                    "test_index": learner_information.test_index.tolist()}

            json_string = json.dumps(data)
            with open(filename + ".map", 'w') as outfile:
                json.dump(json_string, outfile)

            Tools.verbose("Model saved: ("+ filename + ".model, " + filename + ".map)")


    def save_model_generic(self, trees, filename):
        json_string = json.dumps(trees.raw_data())
        with open(filename + ".model", 'w') as outfile:
            json.dump(json_string, outfile)

    def fit_and_predict(self, output, instances_training, instances_test, labels_training, labels_test, learner_options):
        if self.learner_type == LearnerType.Classification:
            if output == EvaluationOutput.DT:
                return self.fit_and_predict_DT_CLS(instances_training, instances_test, labels_training, labels_test, learner_options)
            elif output == EvaluationOutput.RF:
                return self.fit_and_predict_RF_CLS(instances_training, instances_test, labels_training, labels_test, learner_options)
            elif output == EvaluationOutput.BT:
                return self.fit_and_predict_BT_CLS(instances_training, instances_test, labels_training, labels_test, learner_options)
            else:
                raise NotImplementedError(str(output) + " not implemented.")
        elif self.learner_type == LearnerType.Regression:
            if output == EvaluationOutput.DT:
                return self.fit_and_predict_DT_REG(instances_training, instances_test, labels_training, labels_test, learner_options)
            elif output == EvaluationOutput.RF:
                return self.fit_and_predict_RF_REG(instances_training, instances_test, labels_training, labels_test, learner_options)
            elif output == EvaluationOutput.BT:
                return self.fit_and_predict_BT_REG(instances_training, instances_test, labels_training, labels_test, learner_options)
            else:
                raise NotImplementedError(str(output) + " not implemented.")
        else:
            raise NotImplementedError(str(self.learner_type) + " not implemented.")
    

    def hold_out_evaluation(self, output, *, test_size=0.3, learner_options):
        self.learner_information.clear()
        assert self.data is not None, "You have to put the dataset in the class parameters."
        # spliting
        indices = numpy.arange(len(self.data))
        instances_training, instances_test, labels_training, labels_test, training_index, test_index = train_test_split(self.data, self.labels,
                                                                                                                        indices, test_size=test_size,
                                                                                                                        random_state=learner_options["seed"])
        models, metrics, extras = self.fit_and_predict(output, instances_training, instances_test, labels_training, labels_test, learner_options)
        
        self.learner_information.append(LearnerInformation(models, training_index, test_index, None, metrics, extras))

        return self


    def k_folds_evaluation(self, output, *, n_models=10, learner_options):
        assert self.data is not None, "You have to put the dataset in the class parameters."
        assert n_models > 1, "This k_folds_evaluation() expects at least 2 parts. For just one tree, please use hold_out_evaluation()"
        self.learner_information.clear()

        cross_validator = KFold(n_splits=n_models, random_state=learner_options["seed"], shuffle=True)

        for training_index, test_index in cross_validator.split(self.data):
            # Select good observations for each of the 'n_trees' experiments.
            instances_training = [self.data[i] for i in training_index]
            labels_training = [self.labels[i] for i in training_index]
            instances_test = [self.data[i] for i in test_index]
            labels_test = [self.labels[i] for i in test_index]
            
            tree, metrics, extras = self.fit_and_predict(output, instances_training, instances_test, labels_training, labels_test, learner_options)
            
            # Save some information
            self.learner_information.append(LearnerInformation(tree, training_index, test_index, None, metrics, extras))
        return self


    def leave_one_group_out_evaluation(self, output, *, n_trees=10, learner_options):
        assert self.data is not None, "You have to put the dataset in the class parameters."
        assert n_trees > 1, "cross_validation() expects at least 2 trees. For just one tree, please use simple_validation()"
        self.learner_information.clear()

        # spliting
        quotient, remainder = (self.n_instances // n_trees, self.n_instances % n_trees)
        groups = flatten([quotient * [i] for i in range(1, n_trees + 1)]) + [i for i in range(1, remainder + 1)]
        random.Random(learner_options["seed"]).shuffle(groups)
        cross_validator = LeaveOneGroupOut()

        for training_index, test_index in cross_validator.split(self.data, self.labels, groups):
            # Select good observations for each of the 'n_trees' experiments.
            instances_training = [self.data[i] for i in training_index]
            labels_training = [self.labels[i] for i in training_index]
            instances_test = [self.data[i] for i in test_index]
            labels_test = [self.labels[i] for i in test_index]

            # solving
            tree, metrics, extras = self.fit_and_predict(output, instances_training, instances_test, labels_training, labels_test, learner_options)
            
            # Save some information
            self.learner_information.append(LearnerInformation(tree, training_index, test_index, groups, metrics, extras))
        return self


    """
    Return couples (instance, prediction) from data and the classifier results.
    
    'indexes': take only into account some indexes of instances
      - Indexes.Training: indexes from the training instances of a particular model 
      - Indexes.Test: indexes from the test instances of a particular model
      - Indexes.Mixed: take firsly indexes from the test set and next from the training set in order to have at least 'n' instances. 
      - Indexes.All: all indexes are take into account
      - string: A file contening specific indexes 
    
    'dataset': 
      - can be None if the dataset is already loaded
      - the dataset if you have not loaded it yet
      
    'model':
      - a model for the 'type=training' or 'type=test'
        
    'n': The desired number of instances (None for all).

    'correct': only available if a model is given 
      - None: all instances
      - True: only correctly classified instances by the model 
      - False: only misclassified instances by the model 

    'classes': 
      - None: all instances
      - []: List of integers representing the classes/labels for the desired instances  

    'backup_directory': save the instance indexes in a file in the directory given by this parameter 
    """


    def get_instances(self, 
                      model=None, 
                      indexes=Indexes.All, 
                      *, 
                      dataset=None, 
                      n=None, 
                      correct=None, 
                      predictions=None, 
                      save_directory=None,
                      instances_id=None, 
                      seed=0, 
                      training_indexes=None,
                      test_indexes=None, details=False):

        # 1: Check parameters and get the associated solver
        Tools.verbose("---------------   Instances   ----------------")
        assert isinstance(indexes, (Indexes, str)), "Bad value in the parameter 'indexes'"
        if self.get_learner_name() == "Generic":
            assert correct is None, "The 'correct' option is not compatible with a generic model."
            assert predictions is None, "The 'prediction' option is not compatible with a generic model."
        learner, learner_information = self._get_learner(model, indexes, correct, predictions)
        
        # 2: Get the correct indexes:
        possible_indexes = self._get_possible_indexes(indexes, n, instances_id, learner_information, training_indexes, test_indexes)
        
        # 3: Get the correct data (select only data that we need):
        possible_indexes, data, labels = self._get_data(dataset, possible_indexes, n)
        
        # 4: Select instances according to data and possible_indexes.
        instances = []
        instances_indexes = []
        original_indexes = self._build_original_indexes(data, seed)
        instances_details = []
        if model is None or self.get_learner_name() == "Generic":
            for j in original_indexes:
                current_index = possible_indexes[j]
                instances.append((data[j], None))
                instances_indexes.append(current_index)
                instances_details.append({"instance": data[j], "prediction": None, "label": None, "index":current_index})
                if isinstance(n, int) and len(instances) >= n:
                    break
        else:
            for j in original_indexes:
                current_index = possible_indexes[j]
                prediction_solver = learner.predict(numpy.ascontiguousarray(data[j].reshape(1, -1)))[0]
                
                # J'ai, a priori de la chance, que la fonction predict de xgboost et scikit learnt ont la meme def !
                # A voir comment faire, peux être au niveau de extras si on a un probleme avec cela. 
                label = labels[j]
                if (correct and prediction_solver == label) \
                        or (not correct and prediction_solver != label) \
                        or (correct is None):
                    if predictions is None or prediction_solver in predictions:
                        instances.append((data[j], prediction_solver))
                        instances_indexes.append(current_index)
                        instances_details.append({"instance": data[j], "prediction": prediction_solver, "label": label, "index":current_index})
                if isinstance(n, int) and len(instances) >= n:
                    break
        
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
        Tools.verbose("number of instances selected:", len(instances))
        Tools.verbose("----------------------------------------------")
        if details is True:
            if len(instances_details) == 0 and n == 1:
                return None, None
            return instances_details if n is None or n > 1 else instances_details[0] 
        
        if len(instances) == 0 and n == 1:
            return None, None
        return instances if n is None or n > 1 else instances[0]

    def _get_learner(self, model, indexes, correct, predictions):
        if isinstance(model, Iterable):
            raise ValueError("The model does not be an Iterable.")
        elif model is None:
            assert indexes == Indexes.All, "Please insert the model to use this parameter !"
            assert correct is None, "Please insert the model to use this parameter !"
            assert predictions is None, "Please insert the model to use this parameter !"
            return None, None
        elif isinstance(model, BoostedTrees):
            id_learner = model.forest[0].id_solver_results
            learner = self.learner_information[id_learner].raw_model
            learner_information = self.learner_information[id_learner]
            return learner, learner_information
        elif isinstance(model, RandomForest):
            id_learner = model.forest[0].id_solver_results
            learner = self.learner_information[id_learner].raw_model
            learner_information = self.learner_information[id_learner]
            return learner, learner_information
        elif isinstance(model, DecisionTree):
            id_learner = model.id_solver_results
            learner = self.learner_information[id_learner].raw_model
            learner_information = self.learner_information[id_learner]
            return learner, learner_information
        else:
            raise ValueError("The model is not readable: ", str(type(model)))

    def _build_original_indexes(self, data, seed):
        original_indexes = list(range(len(data)))
        if seed is None: 
            random.shuffle(original_indexes)
        elif seed != 0:
            random.Random(seed).shuffle(original_indexes)
        return original_indexes
    
    def _get_possible_indexes(self, indexes, n, instances_id, learner_information, training_indexes, test_indexes):
        if isinstance(indexes, str):
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
                    assert found, "No ." + str(
                        instances_id) + ".instances" + " files in the directory: " + indexes + " Please put directly the good file in the option or use the instances_id parameter !"

            Tools.verbose("Loading instances file:", files_indexes)
            f = open(files_indexes)
            data = json.loads(json.load(f))
            possible_indexes = data['indexes']
            f.close()

        elif indexes == Indexes.Training or indexes == Indexes.Test or indexes == Indexes.Mixed:
            if training_indexes is not None:
                learner_information.training_index = training_indexes
            if test_indexes is not None:
                learner_information.test_index = test_indexes
            if indexes == Indexes.Training and learner_information.training_index is None:
                raise ValueError("The training indexes are unknown: please use the training_indexes parameter.")
            if indexes == Indexes.Test and learner_information.test_index is None:
                raise ValueError("The test indexes are unknown: please use the test_indexes parameter.")
            if indexes == Indexes.Mixed and (learner_information.test_index is None or learner_information.training_index is None):
                raise ValueError("The training or test indexes are unknown: please use the training_indexes or the test_indexes parameters.")
            
            possible_indexes = learner_information.training_index if indexes == Indexes.Training else learner_information.test_index
            if indexes == Indexes.Mixed and n is not None and len(possible_indexes) < n:
                for i in range(n + 1 - len(possible_indexes)):
                    if i < len(learner_information.training_index):
                        possible_indexes = numpy.append(possible_indexes, learner_information.training_index[i])
        else:
            return None        
        
        if isinstance(possible_indexes, numpy.ndarray):
            possible_indexes = possible_indexes.tolist()
        return possible_indexes
    
    def _get_data(self, dataset, possible_indexes, n):
        if self.data is None:
            if dataset is None:
                raise ValueError("Dataset is not loaded yet: you must put the dataset filename through the 'dataset' parameter.")
            if possible_indexes is None:
                data, name = self.parse_data(dataset)
                if data is not None: self.load_data(data, name)
                return [i for i in range(len(self.data))], self.data, self.labels
            else:
                data, labels = self.load_data_limited(dataset, possible_indexes, n)
                return possible_indexes, data, labels
        else:
            if possible_indexes is None:
                possible_indexes = [i for i in range(len(self.data))]
            data = numpy.array([self.data[x] for x in possible_indexes])
            labels = numpy.array([self.labels[x] for x in possible_indexes])
            return possible_indexes, data, labels