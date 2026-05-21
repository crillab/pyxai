from enum import Enum, unique
import os

def auto(n_occurrences=1):
    def _auto():  # To be replaced by auto() in python 3.6 ?
        if not hasattr(auto, "cnt"):
            auto.cnt = 0
        auto.cnt += 1
        return auto.cnt


    return _auto() if n_occurrences == 1 else (_auto() for _ in range(n_occurrences))


@unique
class Encoding(Enum):
    SIMPLE, TSEITIN, COMPLEMENTARY, MUS, SEQUENTIAL_COUNTER, TOTALIZER = auto(6)


    def __str__(self):
        return self.name


@unique
class TypeLeaf(Enum):
    LEFT, RIGHT = auto(2)


    def __str__(self):
        return self.name


# @unique
# class TypeTree(Enum):
#    PREDICTION, WEIGHT = auto(2)
#    
#    def __str__(self):
#        return self.name

@unique
class TypeReason(Enum):
    Direct, Sufficient, MinimalSufficient, Preferred, Contrastive, TreeSpecific, All = auto(7)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name


@unique
class TypeCount(Enum):
    NSufficientReasons, NSufficientReasonsPerAttribute = auto(2)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name




def _check_parameter(parameter: str|Enum|None, enum_type: Enum, argument: str, method:str) -> str|None:
    if parameter is not None:
        parameter = str.lower(parameter)
        if parameter not in [m.value for m in enum_type]:
            str_raise = "'"+parameter+"' is an invalid value for the named argument '" + argument + "' of the method '"+method+"'."+os.linesep
            str_raise += "Here are the possible values of "+str(enum_type)+":"+os.linesep+os.linesep
            for p in enum_type:
                str_raise += str(p) + ': "' + str(p.value) + '"' +os.linesep
            raise ValueError(str_raise) 
    return parameter

class ProblemType(str, Enum):
    Classification = "classification"
    Regression = "regression"

class InstancesType(str, Enum):
    Image = 'image'
    Tabular = 'tabular'
    #Text = 'text'

class LabelsType(str, Enum):
    Classes = "classes"
    Masks = "masks"
    #Texts = "texts"
    DiscreteValues = "discrete-values"
    ContinuousValues = "continuous-values"
    Contours = "contours"

class SplittingMethod(str, Enum):
    Predefined = "predefined"
    LoadModel = "load-model"
    HoldOut = "hold-out"
    LeaveOneGroupOut = "leave-one-group-out"
    KFolds = "k-folds"
        
class ModelType(str, Enum):
    LinearSimple = "linear-simple"
    LinearRidge = "linear-ridge"
    LinearLasso = "linear-lasso"
    LinearElastic = "linear-elastic"
    LinearLogistic = "linear-logistic"
    NeuralNetwork = "neural-network"
    DecisionTree = "decision-tree"
    RandomForest = "random-forest"
    BoostedTree = "boosted-tree"
    Save = "save"


class InstancesSet(str, Enum):
    Train = "train"
    Test = "test"
    TrainInPriority = "train-in-priority"
    All = "all"


class PreferredReasonMethod(str, Enum):
    Minimal = "minimal"
    Shapley = "shapley"
    FeatureImportance = "feature-importance"
    WordFrequency = "word-frequency"
    WordFrequencyLayers = "word-frequency-layers"
    Weights = "weights"
    InclusionPreferred = "inclusion-preferred"


class ReasonExpressivity(str, Enum):
    Features = "features"
    Conditions = "conditions"

    


class SaveFormat(str, Enum):
    SolverSpecific = "solver-specific"
    RawData = "raw-data"


class TypeStatus(str, Enum):
    Unsat = "unsat"
    Sat = "sat"
    Optimum = "optimum"
    Core = "core"
    Unknown = "unknown"


class OperatorCondition(str, Enum):
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"

    def to_str_readable(self):
        if self.value == OperatorCondition.EQ:
            return "=="
        if self.value == OperatorCondition.GE:
            return ">="
        if self.value == OperatorCondition.GT:
            return ">"
        if self.value == OperatorCondition.LE:
            return "<="
        if self.value == OperatorCondition.LT:
            return "<"
        if self.value == OperatorCondition.EQ:
            return "="
        if self.value == OperatorCondition.NEQ:
            return "!="
        raise NotImplementedError("The operator " + str(self.value) + " is not implemented.")


class TypeTheory(str, Enum):
    Simple = "simple"
    NewVariables = "new-variables"


class TypeFeature(str, Enum):
    Numerical = "numerical"
    Categorical = "categorical"
    Binary = "binary" 
    Target = "target"
    To_delete = "to_delete"
    Default = "default"

class ClassificationType(str, Enum):
    BinaryClass = "binary-class"
    MultiClass = "multi-class"


class MultiClassToBinaryMethod(str, Enum):
    OneVsRest = "one-vs-rest"
    OneVsOne = "one-vs-one"


class EncoderType(str, Enum):
    OrdinalEncoder = "ordinal-encoder"
    OneHotEncoder = "oneHot-encoder"
    DiscretizerEncoder = "discretizer-encoder"

