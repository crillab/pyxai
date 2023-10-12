import json
from operator import index
from typing import Iterable

from pyxai.sources.core.structure.type import EvaluationMethod, EvaluationOutput, Indexes, SaveFormat, TypeFeature, TypeClassification, MethodToBinaryClassification, TypeEncoder, LearnerType
from pyxai.sources.learning.learner import LearnerInformation, Learner, NoneData
from pyxai.sources.learning.generic import Generic
from pyxai.sources.learning.scikitlearn import Scikitlearn
from pyxai.sources.learning.xgboost import Xgboost
from pyxai.sources.learning.lightgbm import LightGBM
from pyxai.sources.learning.preprocessor import Preprocessor
from pyxai import Tools


HOLD_OUT = EvaluationMethod.HoldOut
LEAVE_ONE_GROUP_OUT = EvaluationMethod.LeaveOneGroupOut
K_FOLDS = EvaluationMethod.KFolds

DT = EvaluationOutput.DT
RF = EvaluationOutput.RF
BT = EvaluationOutput.BT
SAVE = EvaluationOutput.SAVE

CLASSIFICATION = LearnerType.Classification
REGRESSION = LearnerType.Regression

TRAINING = Indexes.Training
TEST = Indexes.Test
MIXED = Indexes.Mixed
ALL = Indexes.All

RAW_DATA = SaveFormat.RawData
SOLVER_SPECIFIC = SaveFormat.SolverSpecific

NUMERICAL = TypeFeature.NUMERICAL
CATEGORICAL = TypeFeature.CATEGORICAL
DEFAULT = TypeFeature.DEFAULT

BINARY_CLASS = TypeClassification.BinaryClass
MULTI_CLASS = TypeClassification.MultiClass

ONE_VS_REST = MethodToBinaryClassification.OneVsRest
ONE_VS_ONE = MethodToBinaryClassification.OneVsOne

ORDINAL = TypeEncoder.OrdinalEncoder
ONE_HOT = TypeEncoder.OneHotEncoder

def import_models(models, feature_names=None):
        
    if not isinstance(models, (list, tuple)):
        models = [models]
    if not all(type(model) == type(models[0]) for model in models):
        raise ValueError("All models must be of the same type: " + str(type(models[0])))
    
    if type(models[0]) in Scikitlearn.get_learner_types().keys():
        learner_type = Scikitlearn.get_learner_types()[type(models[0])][0]
        evaluation_output = Scikitlearn.get_learner_types()[type(models[0])][1]
        learner = Scikitlearn(NoneData, learner_type=learner_type)
        if learner_type == CLASSIFICATION:
            learner.create_dict_labels(models[0].classes_)
            learner.labels = learner.labels_to_values(models[0].classes_)
            learner.n_labels = len(set(learner.labels))
        extras = {
            "learner": str(type(models[0])),
            "learner_options": models[0].get_params(),
            "base_score": 0,
        }
    elif type(models[0]) in Xgboost.get_learner_types().keys():
        learner_type = Xgboost.get_learner_types()[type(models[0])][0]
        evaluation_output = Xgboost.get_learner_types()[type(models[0])][1]
        learner = Xgboost(NoneData, learner_type=learner_type)
        extras = {
            "learner": str(type(models[0])),
            "learner_options": models[0].get_xgb_params(),
            "base_score": float(0.5) if models[0].base_score is None else models[0].base_score,
        }
            
        if learner_type == CLASSIFICATION:
            learner.create_dict_labels(models[0].classes_)
            learner.labels = learner.labels_to_values(models[0].classes_)
            learner.n_labels = len(set(learner.labels))
    elif type(models[0]) in LightGBM.get_learner_types().keys():
        
        learner_type = LightGBM.get_learner_types()[type(models[0])][0]
        evaluation_output = LightGBM.get_learner_types()[type(models[0])][1]
        learner = LightGBM(NoneData, learner_type=learner_type)
        extras = {
            "learner": str(type(learner)),
            "learner_options": models[0].get_params(),
            "base_score": 0
        }
           
        if learner_type == CLASSIFICATION:
            learner.create_dict_labels(models[0].classes_)
            learner.labels = learner.labels_to_values(models[0].classes_)
            learner.n_labels = len(set(learner.labels))
    else:
        raise ValueError("The type of this model is unknown: "+str(type(models[0])))

    learner_information=[LearnerInformation(model, None, None, None, None, extras) for model in models]
    if feature_names is not None:
        learner.feature_names = feature_names
        for l in learner_information: l.set_feature_names(feature_names)
    
    result_output = learner.convert_model(evaluation_output, learner_information)
    
    Tools.verbose("---------------   Explainer   ----------------")
    for i, result in enumerate(result_output):
        Tools.verbose("For the evaluation number " + str(i) + ":")
        Tools.verbose(result)

    return learner, result_output if len(result_output) != 1 else result_output[0]




def load(models_directory, *,tests=False, dataset=NoneData):
    learner = Learner(learner_type=CLASSIFICATION)
    files = learner.load_get_files(models_directory)
    learner_names = []
    learner_types = []
    evaluation_outputs = []

    for _, model in enumerate(files):
        _, map_file = model
        f = open(map_file)
        data = json.loads(json.load(f))
        learner_names.append(data['learner_name'])
        learner_types.append(data['learner_type'])
        evaluation_outputs.append(data['evaluation_output'])
        f.close()
    
    if not all(learner_name == learner_names[-1] for learner_name in learner_names):
        raise ValueError("All learners must have the same learner name.")
    if not all(learner_type == learner_types[-1] for learner_type in learner_types):
        raise ValueError("All learners must have the same learner type.")
    if not all(evaluation_output == evaluation_outputs[-1] for evaluation_output in evaluation_outputs):
        raise ValueError("All learners must have the same learner type.")
    
    if learner_names[0] == Generic.__name__:
        learner = Generic(dataset, learner_type=LearnerType.from_str(learner_types[0]))
    elif learner_names[0] == Xgboost.__name__:
        learner = Xgboost(dataset, learner_type=LearnerType.from_str(learner_types[0]))
    elif learner_names[0] == Scikitlearn.__name__:
        learner = Scikitlearn(dataset, learner_type=LearnerType.from_str(learner_types[0]))
    elif learner_names[0] == LightGBM.__name__:
        learner = LightGBM(dataset, learner_type=LearnerType.from_str(learner_types[0]))
    else:
        raise ValueError("The learner name is unknown:" + str(learner_names[0]))

    models = learner.load(models_directory=models_directory, tests=tests)
    return learner, models
