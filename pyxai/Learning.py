
from pyxai.sources.type import _check_parameter, SplittingMethod, InstancesSet, SaveFormat, TypeFeature, ClassificationType, MultiClassToBinaryMethod, EncoderType
from pyxai.sources.type import ProblemType, ModelType, InstancesType, LabelsType
from pyxai.sources.learners.learner import Learner, NoneData
from pyxai.sources.learners.learner_information import LearnerInformation
from pyxai.sources.learners.scikitlearn import Scikitlearn
from pyxai.sources.learners.xgboost import Xgboost
from pyxai.sources.learners.lightgbm import LightGBM

from pyxai.sources.learners.preprocessor.tabular_preprocessor import TabularPreprocessor
from pyxai.sources.learners.preprocessor.non_tabular_preprocessor import NonTabularPreprocessor

from pyxai.sources.learners.model_io import ModelIO

#Cross-Validation methods
PREDEFINED = SplittingMethod.Predefined
HOLD_OUT = SplittingMethod.HoldOut
LEAVE_ONE_GROUP_OUT = SplittingMethod.LeaveOneGroupOut
K_FOLDS = SplittingMethod.KFolds

#Regression and Classification
DT = ModelType.DecisionTree
RF = ModelType.RandomForest
BT = ModelType.BoostedTree

#Linear Regression Type 
LINEAR_SIMPLE = ModelType.LinearSimple
LINEAR_LASSO = ModelType.LinearLasso
LINEAR_ELASTIC = ModelType.LinearElastic

#Linear Classification Type
LINEAR_RIDGE = ModelType.LinearRidge
LINEAR_LOGISTIC = ModelType.LinearLogistic
NEURAL_NETWORK = ModelType.NeuralNetwork


#Save
SAVE = ModelType.Save

#Type
CLASSIFICATION = ProblemType.Classification
REGRESSION = ProblemType.Regression

#Selecting Indexes
TRAIN = InstancesSet.Train
TEST = InstancesSet.Test
TRAIN_IN_PRIORITY  = InstancesSet.TrainInPriority 
ALL = InstancesSet.All

#Save format
RAW_DATA = SaveFormat.RawData
SOLVER_SPECIFIC = SaveFormat.SolverSpecific

#Feature types
NUMERICAL = TypeFeature.Numerical
CATEGORICAL = TypeFeature.Categorical
DEFAULT = TypeFeature.Default

#Classification types
BINARY_CLASS = ClassificationType.BinaryClass
MULTI_CLASS = ClassificationType.MultiClass

#Convertion Multi-classe methods
ONE_VS_REST = MultiClassToBinaryMethod.OneVsRest
ONE_VS_ONE = MultiClassToBinaryMethod.OneVsOne

#Convertion feature types
ORDINAL = EncoderType.OrdinalEncoder
ONE_HOT = EncoderType.OneHotEncoder
DISCRETIZER = EncoderType.DiscretizerEncoder

#Type of instances
IMAGE = InstancesType.Image
TABULAR = InstancesType.Tabular

#Type of labels
CLASSES = LabelsType.Classes
MASKS = LabelsType.Masks
CONTOURS = LabelsType.Contours










