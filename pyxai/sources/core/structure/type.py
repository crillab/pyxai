from enum import Enum, unique


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


@unique
class EvaluationMethod(Enum):
    LoadModel, HoldOut, LeaveOneGroupOut, KFolds = auto(4)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name


class LearnerType(Enum):
    Classification, Regression = auto(2)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name


    def from_str(str):
        if str == "Classification":
            return LearnerType.Classification
        elif str == "Regression":
            return LearnerType.Regression
        else:
            assert False, "No EvaluationOutput for this string !"

@unique
class EvaluationOutput(Enum):
    DT, RF, BT, SAVE = auto(4)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name


    def from_str(str):
        if str == "DT":
            return EvaluationOutput.DT
        elif str == "RF":
            return EvaluationOutput.RF
        elif str == "BT":
            return EvaluationOutput.BT
        else:
            assert False, "No EvaluationOutput for this string !"


class Indexes(Enum):
    Training, Test, Mixed, All = auto(4)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name


class PreferredReasonMethod(Enum):
    Minimal, Shapley, FeatureImportance, WordFrequency, WordFrequencyLayers, Weights, InclusionPreferred = auto(7)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name


@unique
class ReasonExpressivity(Enum):
    Features, Conditions = auto(2)


    def __eq__(self, other):
        if type(self) != type(other):
            raise TypeError("Bad type for : " + other + " (" + str(type(other)) + "). Must be of the type " + str(type(self)) + ".")
        return self.value == other.value


    def __int__(self):
        if self == ReasonExpressivity.Features:
            return 1
        elif self == ReasonExpressivity.Conditions:
            return 0


    def __str__(self):
        return self.name


@unique
class SaveFormat(Enum):
    SolverSpecific, RawData = auto(2)


    def __eq__(self, other):
        return self.value == other.value


    def __str__(self):
        return self.name


@unique
class TypeStatus(Enum):
    UNSAT, SAT, OPTIMUM, CORE, UNKNOWN = auto(5)


    def __str__(self):
        return self.name

@unique
class OperatorCondition(Enum):
    EQ, NEQ, LT, LE, GT, GE = auto(6)


    def __hash__(self) -> int:
        return hash(self.__str__())


    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        
        return self.value == other.value

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
    
    def __str__(self):
        return self.name

@unique
class TypeTheory(Enum):
    SIMPLE, NEW_VARIABLES = auto(2)

    def __str__(self):
        return self.name

@unique
class TypeFeature(Enum):
    NUMERICAL, CATEGORICAL, BINARY, TARGET, TO_DELETE, DEFAULT = auto(6)

    def __str__(self):
        return self.name

    def from_str(str):
        if str == "CATEGORICAL":
            return TypeFeature.CATEGORICAL
        elif str == "NUMERICAL":
            return TypeFeature.NUMERICAL
        elif str == "BINARY":
            return TypeFeature.BINARY
        elif str == "TARGET":
            return TypeFeature.TARGET
        elif str == "TO_DELETE":
            return TypeFeature.TO_DELETE
        elif str == "DEFAULT":
            return TypeFeature.DEFAULT
        else:
            assert False, "No TypeFeature for this string !"


@unique
class TypeClassification(Enum):
    BinaryClass, MultiClass = auto(2)

    def __str__(self):
        return self.name

@unique
class MethodToBinaryClassification(Enum):
    OneVsRest, OneVsOne = auto(2)

    def __str__(self):
        return self.name

@unique
class TypeEncoder(Enum):
    OrdinalEncoder, OneHotEncoder = auto(2)

    def __str__(self):
        return self.name
    
    def from_str(str):
        if str == "OrdinalEncoder":
            return TypeEncoder.OrdinalEncoder
        elif str == "OneHotEncoder":
            return TypeEncoder.OneHotEncoder
        else:
            assert False, "No TypeEncoder for this string !"