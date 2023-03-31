import json
from operator import index

from pyxai.sources.core.structure.type import EvaluationMethod, EvaluationOutput, Indexes, SaveFormat, TypeFeature, TypeClassification, MethodToBinaryClassification, TypeEncoder, LearnerType
from pyxai.sources.learning.Learner import LearnerInformation, Learner, NoneData
from pyxai.sources.learning.generic import Generic
from pyxai.sources.learning.scikitlearn import Scikitlearn
from pyxai.sources.learning.xgboost import Xgboost
from pyxai.sources.learning.lightgbm import LightGBM
from pyxai.sources.learning.converter import Converter


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
        learner = Generic(dataset)
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
