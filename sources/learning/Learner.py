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
from pyxai.sources.core.structure.type import EvaluationMethod, EvaluationOutput, Indexes
from pyxai.sources.core.tools.utils import flatten, compute_accuracy


class LearnerInformation:
    def __init__(self, raw_model, training_index=None, test_index=None, group=None, accuracy=None):
        self.raw_model = raw_model
        self.training_index = training_index
        self.test_index = test_index
        self.group = group
        self.accuracy = accuracy
        self.solver_name = None
        self.feature_names = None
        self.evaluation_method = None
        self.evaluation_output = None


    def set_solver_name(self, solver_name):
        self.solver_name = solver_name


    def set_feature_names(self, feature_names):
        self.feature_names = feature_names


    def set_evaluation_method(self, evaluation_method):
        self.evaluation_method = str(evaluation_method)


    def set_evaluation_output(self, evaluation_output):
        self.evaluation_output = str(evaluation_output)


class NoneData:
    pass


class Learner:
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    instance = observation 
    labels != prediction
    attributes = features
    """


    def __init__(self, data=NoneData):
        self.dict_labels = None
        self.data = None
        self.labels = None
        self.n_labels = None
        self.n_features = None
        self.dataset_name = None
        self.n_instances = None
        self.feature_names = None
        self.learner_information = []
        data, name = self.parse_data(data)
        if data is not None:
            self.load_data(data, name)


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
            raise ValueError("The data parameter is set but None (can be optional, but not None).")

        raise ValueError("The data parameter is either a string representing a csv or excel file or a pandas.core.frame.DataFrame object.")


    @staticmethod
    def count_lines(filename):
        with open(filename) as f:
            return sum(1 for _ in f)


    def load_data_limited(self, datasetname, possibles_indexes, n):
        self.dataset_name = datasetname
        n_indexes = self.count_lines(datasetname) - 1  # to skip the first line

        skip = [i + 1 for i in range(n_indexes) if i not in possibles_indexes] if possibles_indexes is not None else None

        # create a map to get the good order of instances
        if skip is not None:
            sorted_possibles_indexes = sorted(possibles_indexes)
            map_possibles_indexes = [sorted_possibles_indexes.index(index) for index in possibles_indexes]

        data = pandas.read_csv(
            datasetname,
            skiprows=skip,
            nrows=n
        )

        # recreate the dataframe object but with the good order of instances
        if skip is not None:
            sorted_data = pandas.DataFrame(columns=data.columns).astype(data.dtypes)
            for i in range(data.shape[0]):
                sorted_data = sorted_data.append(data.loc[map_possibles_indexes[i]].to_dict(), ignore_index=True)
            sorted_data = sorted_data.astype(data.dtypes)

        n_instances, n_features = data.shape
        self.feature_names = data.columns.values.tolist()
        self.rename_attributes(data)
        data, labels = self.remove_labels(data, n_features)
        labels = self.labels_to_values(labels)
        data = data.to_numpy()

        return data, labels


    def load_data(self, dataframe, datasetname):
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
        self.create_dict_labels(self.labels)
        self.labels = self.labels_to_values(self.labels)

        self.n_labels = len(set(self.labels))
        self.data = self.data.to_numpy()  # remove the first line (attributes) and now the first dimension represents the instances :)!
        self.learner_information = []
        Tools.verbose("---------------   Information   ---------------")
        Tools.verbose("Dataset name:", self.dataset_name)
        Tools.verbose("nFeatures (nAttributes, with the labels):", self.n_features)
        Tools.verbose("nInstances (nObservations):", self.n_instances)
        Tools.verbose("nLabels:", self.n_labels)


    """
    Rename attributes in self.data in string of integers from 0 to 'self.n_attributes'
    """


    @staticmethod
    def rename_attributes(data):
        rename_dictionary = {element: str(i) for i, element in enumerate(data.columns)}
        data.rename(columns=rename_dictionary, inplace=True)


    def create_dict_labels(self, labels):
        index = 0
        self.dict_labels = OrderedDict()
        for p in labels:
            if str(p) not in self.dict_labels:
                self.dict_labels[str(p)] = index
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


    """
    Method:

    """


    def evaluate(self, *, method, output, n_models=10, test_size=0.3, max_depth=None, seed=0, model_directory=None):
        Tools.verbose("---------------   Evaluation   ---------------")
        Tools.verbose("method:", str(method))
        Tools.verbose("output:", str(output))

        if method == EvaluationMethod.HoldOut:
            self.hold_out_evaluation(output, test_size=test_size, max_depth=max_depth, seed=seed)
        elif method == EvaluationMethod.LeaveOneGroupOut:
            self.leave_one_group_out_evaluation(output, n_trees=n_models, max_depth=max_depth, seed=seed)
        elif method == EvaluationMethod.KFolds:
            self.k_folds_evaluation(output, n_models=n_models, max_depth=max_depth, seed=seed)
        else:
            assert False, "Not implemented !"

        for learner_information in self.learner_information:
            learner_information.set_solver_name(self.get_solver_name())
            learner_information.set_feature_names(self.feature_names)
            learner_information.set_evaluation_method(method)
            learner_information.set_evaluation_output(output)

        Tools.verbose("---------   Evaluation Information   ---------")
        for i, result in enumerate(self.learner_information):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose("accuracy:", result.accuracy)
            Tools.verbose("nTraining instances:", len(result.training_index))
            Tools.verbose("nTest instances:", len(result.test_index))
            Tools.verbose()

        Tools.verbose("---------------   Explainer   ----------------")
        result_output = None
        if output == EvaluationOutput.DT:
            result_output = self.to_DT(self.learner_information)
        elif output == EvaluationOutput.RF:
            result_output = self.to_RF(self.learner_information)
        elif output == EvaluationOutput.BT:
            result_output = self.to_BT(self.learner_information)
        else:
            assert False, "Not implemented !"
        # elif output == EvaluationOutput.SAVE:
        #  self.save_model(model_directory)
        #  result_output = self.to_BT()

        for i, result in enumerate(result_output):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose(result)
        return result_output if len(result_output) != 1 else result_output[0]


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


    def load(self, *, models_directory, with_tests=False):
        files = self.load_get_files(models_directory)

        for _, model in enumerate(files):
            model_file, map_file = model

            # recuperate map
            f = open(map_file)
            data = json.loads(json.load(f))
            training_index = data['training_index']
            test_index = data['test_index']
            accuracy_saved = data['accuracy']
            solver_name = data['solver_name']
            evaluation_method = data['evaluation_method']
            evaluation_output = data['evaluation_output']

            self.n_features = data['n_features']
            self.n_labels = data["n_labels"]
            self.dict_labels = data["dict_labels"]
            self.feature_names = data["feature_names"]

            f.close()

            assert self.get_solver_name() == solver_name, "You try to load a model with " + self.get_solver_name() + ", but you have save the model with " + solver_name + " !"

            # load model
            classifier = self.load_model(model_file)

            Tools.verbose("----------   Loading Information   -----------")
            Tools.verbose("mapping file:", map_file)
            Tools.verbose("nFeatures (nAttributes, with the labels):", self.n_features)
            Tools.verbose("nInstances (nObservations):", len(training_index) + len(test_index))
            Tools.verbose("nLabels:", self.n_labels)
            if with_tests:
                # Test phase
                instances_test = [self.data[i] for i in test_index]
                labels_test = [self.labels[i] for i in test_index]
                result = classifier.predict(instances_test)
                accuracy = compute_accuracy(result, labels_test)
                assert accuracy == accuracy_saved, "The accuracy between the model loaded and the one determined at its creation is not the same !"
            self.learner_information.append(LearnerInformation(copy.deepcopy(classifier), training_index, test_index, None, accuracy_saved))
            self.learner_information[-1].set_solver_name(solver_name)
            self.learner_information[-1].set_evaluation_method(evaluation_method)
            self.learner_information[-1].set_evaluation_output(evaluation_output)
            self.learner_information[-1].set_feature_names(self.feature_names)

        assert all(learner_information.evaluation_output == self.learner_information[-1].evaluation_output for learner_information in
                   self.learner_information), "All evaluation outputs have to be the same !"

        Tools.verbose("---------   Evaluation Information   ---------")
        for i, result in enumerate(self.learner_information):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose("accuracy:", result.accuracy)
            Tools.verbose("nTraining instances:", len(result.training_index))
            Tools.verbose("nTest instances:", len(result.test_index))
            Tools.verbose()

        Tools.verbose("---------------   Explainer   ----------------")
        output = EvaluationOutput.from_str(self.learner_information[-1].evaluation_output)
        result_output = None

        if output == EvaluationOutput.DT:
            result_output = self.to_DT(self.learner_information)
        elif output == EvaluationOutput.RF:
            result_output = self.to_RF(self.learner_information)
        elif output == EvaluationOutput.BT:
            result_output = self.to_BT(self.learner_information)
        else:
            assert False, "Not implemented !"

        for i, result in enumerate(result_output):
            Tools.verbose("For the evaluation number " + str(i) + ":")
            Tools.verbose(result)
        return result_output if len(result_output) != 1 else result_output[0]


    def save(self, models, save_directory, generic=False):

        name = self.dataset_name.split(os.sep)[-1].split('.')[0]
        if save_directory is not None:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            base_directory = save_directory
        else:
            base_directory = ""

        shutil.rmtree(base_directory, ignore_errors=True)
        os.mkdir(base_directory)

        if not isinstance(models, Iterable):
            models = [models]

        for i, trees in enumerate(models):
            learner_information = trees.learner_information
            filename = base_directory + os.sep + name + '.' + str(i)
            # model:
            if not generic:
                self.save_model(learner_information, filename)
            else:

                self.save_model_generic(trees, filename)
            # map of indexes for training and test part
            data = {"training_index": learner_information.training_index.tolist(),
                    "test_index": learner_information.test_index.tolist(),
                    "accuracy": learner_information.accuracy,
                    "solver_name": learner_information.solver_name if not generic else "Generic",
                    "evaluation_method": learner_information.evaluation_method,
                    "evaluation_output": learner_information.evaluation_output,
                    "format": str(format),

                    "n_features": self.n_features,
                    "n_labels": self.n_labels,
                    "dict_labels": self.dict_labels,
                    "feature_names": self.feature_names}

            json_string = json.dumps(data)
            with open(filename + ".map", 'w') as outfile:
                json.dump(json_string, outfile)

            Tools.verbose("Model saved:", filename + ".model")
            Tools.verbose("Model saved:", filename + ".map")


    def save_model_generic(self, trees, filename):
        json_string = json.dumps(trees.raw_data())
        with open(filename + ".model", 'w') as outfile:
            json.dump(json_string, outfile)


    def hold_out_evaluation(self, output, *, seed=0, max_depth=None, test_size=0.3):
        self.learner_information.clear()
        assert self.data is not None, "You have to put the dataset in the class parameters."
        # spliting
        indices = numpy.arange(len(self.data))
        instances_training, instances_test, labels_training, labels_test, training_index, test_index = train_test_split(self.data, self.labels,
                                                                                                                        indices, test_size=test_size,
                                                                                                                        random_state=seed)

        # solving
        if output == EvaluationOutput.DT:
            tree, accuracy = self.fit_and_predict_DT(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth, seed=seed)
        elif output == EvaluationOutput.RF:
            tree, accuracy = self.fit_and_predict_RF(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth, seed=seed)
        elif output == EvaluationOutput.BT:
            tree, accuracy = self.fit_and_predict_BT(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth, seed=seed)
        else:
            assert False, "hold_out_evaluation: EvaluationOutput Not implemented !"
        self.learner_information.append(LearnerInformation(tree, training_index, test_index, None, accuracy))
        return self


    def k_folds_evaluation(self, output, *, n_models=10, max_depth=None, seed=0):
        assert self.data is not None, "You have to put the dataset in the class parameters."
        assert n_models > 1, "This k_folds_evaluation() expects at least 2 parts. For just one tree, please use hold_out_evaluation()"
        self.learner_information.clear()

        cross_validator = KFold(n_splits=n_models, random_state=seed, shuffle=True)

        for training_index, test_index in cross_validator.split(self.data):
            # Select good observations for each of the 'n_trees' experiments.
            instances_training = [self.data[i] for i in training_index]
            labels_training = [self.labels[i] for i in training_index]
            instances_test = [self.data[i] for i in test_index]
            labels_test = [self.labels[i] for i in test_index]

            # solving
            if output == EvaluationOutput.DT:
                tree, accuracy = self.fit_and_predict_DT(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth,
                                                         seed=seed)
            elif output == EvaluationOutput.RF:
                tree, accuracy = self.fit_and_predict_RF(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth,
                                                         seed=seed)
            elif output == EvaluationOutput.BT:
                tree, accuracy = self.fit_and_predict_BT(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth,
                                                         seed=seed)
            else:
                assert False, "leave_one_group_out_evaluation: EvaluationOutput Not implemented !"

            # Save some information
            self.learner_information.append(LearnerInformation(tree, training_index, test_index, None, accuracy))
        return self


    def leave_one_group_out_evaluation(self, output, *, n_trees=10, max_depth=None, seed=0):
        assert self.data is not None, "You have to put the dataset in the class parameters."
        assert n_trees > 1, "cross_validation() expects at least 2 trees. For just one tree, please use simple_validation()"
        self.learner_information.clear()

        # spliting
        quotient, remainder = (self.n_instances // n_trees, self.n_instances % n_trees)
        groups = flatten([quotient * [i] for i in range(1, n_trees + 1)]) + [i for i in range(1, remainder + 1)]
        random.Random(seed).shuffle(groups)
        cross_validator = LeaveOneGroupOut()

        for training_index, test_index in cross_validator.split(self.data, self.labels, groups):
            # Select good observations for each of the 'n_trees' experiments.
            instances_training = [self.data[i] for i in training_index]
            labels_training = [self.labels[i] for i in training_index]
            instances_test = [self.data[i] for i in test_index]
            labels_test = [self.labels[i] for i in test_index]

            # solving
            # solving
            if output == EvaluationOutput.DT:
                tree, accuracy = self.fit_and_predict_DT(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth,
                                                         seed=seed)
            elif output == EvaluationOutput.RF:
                tree, accuracy = self.fit_and_predict_RF(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth,
                                                         seed=seed)
            elif output == EvaluationOutput.BT:
                tree, accuracy = self.fit_and_predict_BT(instances_training, instances_test, labels_training, labels_test, max_depth=max_depth,
                                                         seed=seed)
            else:
                assert False, "leave_one_group_out_evaluation: EvaluationOutput Not implemented !"

            # Save some information
            self.learner_information.append(LearnerInformation(tree, training_index, test_index, groups, accuracy))
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


    def get_instances(self, model=None, indexes=Indexes.All, *, dataset=None, n=None, correct=None, predictions=None, save_directory=None,
                      instances_id=None, seed=0):

        # 1: Check parameters and get the associated solver
        Tools.verbose("---------------   Instances   ----------------")
        #Tools.verbose("Correctness of instances : ", correct)
        #Tools.verbose("Predictions of instances : ", predictions)
        classifier = None
        id_solver_results = None
        results = None

        assert isinstance(indexes, (Indexes, str)), "Bad value in the parameter 'indexes'"

        if self.get_solver_name() == "Generic":
            assert correct is None, "Please insert the model to use this parameter !"
            assert predictions is None, "Please insert the model to use this parameter !"

        if model is None:
            assert indexes == Indexes.All, "Please insert the model to use this parameter !"
            assert correct is None, "Please insert the model to use this parameter !"
            assert predictions is None, "Please insert the model to use this parameter !"
            # In this case, no prediction, just return some instances
        elif isinstance(model, Iterable):
            assert False, "The model is not a model !"
        else:
            # depending of the model
            if isinstance(model, BoostedTrees):
                id_solver_results = model.forest[0].id_solver_results
                classifier = self.learner_information[id_solver_results].raw_model
                results = self.learner_information[id_solver_results]
            if isinstance(model, RandomForest):
                id_solver_results = model.forest[0].id_solver_results
                classifier = self.learner_information[id_solver_results].raw_model
                results = self.learner_information[id_solver_results]
            if isinstance(model, DecisionTree):
                id_solver_results = model.id_solver_results
                classifier = self.learner_information[id_solver_results].raw_model
                results = self.learner_information[id_solver_results]

        # 2: Get the correct indexes:
        possible_indexes = None

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
            possible_indexes = results.training_index if indexes == Indexes.Training else results.test_index
            if indexes == Indexes.Mixed and n is not None and len(possible_indexes) < n:
                for i in range(n + 1 - len(possible_indexes)):
                    if i < len(results.training_index):
                        possible_indexes = numpy.append(possible_indexes, results.training_index[i])
        # Tools.verbose("possible indexes:", possible_indexes, n)
        # load data and get instances
        # 2b : shuffle data if asked
        if isinstance(possible_indexes, numpy.ndarray):
            possible_indexes = possible_indexes.tolist()

        # 3: Get the correct data (select only data that we need):
        if self.data is None:
            assert dataset is not None, "Data are not loaded yet. You have to put your dataset filename through the 'dataset' parameter !"
            if possible_indexes is not None:
                data, labels = self.load_data_limited(dataset, possible_indexes, n)
            else:
                possible_indexes = [i for i in range(len(self.data))]
                data, name = self.parse_data(dataset)
                if data is not None:
                    self.load_data(data, name)
                data = self.data
                labels = self.labels
        else:
            if possible_indexes is None:
                possible_indexes = [i for i in range(len(self.data))]
                data = [self.data[x] for x in possible_indexes]
                labels = self.labels
            else:
                data = numpy.array([self.data[x] for x in possible_indexes])
                labels = numpy.array([self.labels[x] for x in possible_indexes])

        # 4: Select instances according to parameters and data that is modify with only instances of possible_indexes.
        instances = []
        instances_indexes = []

        original_indexes = list(range(len(data)))
        if seed is not None:
          random.Random(seed).shuffle(original_indexes)
        else:
            random.shuffle(original_indexes)

        if model is None or self.get_solver_name() == "Generic":
            for j in original_indexes:
                current_index = possible_indexes[j]
                instances.append((data[j], None))
                instances_indexes.append(current_index)
                if isinstance(n, int) and len(instances) >= n:
                    break
        else:
            for j in original_indexes:
                current_index = possible_indexes[j]

                prediction_solver = classifier.predict(data[j].reshape(1, -1))[
                    0]  # J'ai, a priori de la chance, que la fonction predict de xgboost et scikit learnt ont la meme def !
                label = labels[j]
                if (correct and prediction_solver == label) \
                        or (not correct and prediction_solver != label) \
                        or (correct is None):
                    if predictions is None or prediction_solver in predictions:
                        instances.append((data[j], prediction_solver))
                        instances_indexes.append(current_index)
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
        if len(instances) == 0 and n == 1:
            return None, None
        return instances if n is None or n > 1 else instances[0]
