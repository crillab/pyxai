
import os
import numpy

from pyxai.sources.data_structures.baseModel import BaseModel

from sklearn.neural_network._base import ACTIVATIONS

class LinearModel(BaseModel):
    """This classe is the parent class of a linear model. """
    
    def __init__(self, n_features, coefficients, intercepts, n_classes=None, learner_information=None, feature_names=None, class_names=None):
        super().__init__(n_features, n_classes, learner_information, feature_names, class_names)
        self.coefficients = coefficients
        self.intercepts = intercepts
        #self.coefficients = coefficients if isinstance(coefficients, tuple) else tuple(coefficients, )
        #if isinstance(intercepts, tuple): self.intercepts = intercepts
        #elif type(intercepts)==numpy.float64: self.intercepts =  tuple([intercepts], )
        #else: self.intercepts = tuple(intercepts, )
        
    def __str__(self):
        s = "**Linear Model**" + os.linesep
        s += "nFeatures: " + str(self.n_features) + os.linesep
        s += "Coefficient: " + str(self.coefficients) + os.linesep
        s += "Intercept: " + str(self.intercepts) + os.linesep
        return s

    def get_used_features(self):
        return self.n_features

    def raw_data_for_CPP(self):
        raise NotImplementedError

    def raw_data(self):
        raise NotImplementedError

    def direct_reason(self, instance):
        raise NotImplementedError

    def _compute_id_binaries(self, force_features_equal_to_binaries=False):
        return None, None

    def _is_implicant(self, reason, target_prediction):
        raise NotImplementedError

    def instance_to_binaries(self, instance):
        return NotImplementedError

    def _predict_instance(self, instance):
        return instance @ self.coefficients + self.intercepts
        """
        Return the prediction (the classification) of an observation (instance) according to the model
        """
        
        #return self.intercepts[0] + sum(instance[i]*self.coefficients[i] for i in range(len(instance)))
        
    def _predict_implicant(self, implicant):
        raise NotImplementedError

    def _take_decisions_binary_representation(self, binary_representation):
        """
        Return the prediction (the classification) of an binary representation according to this model
        """
        raise NotImplementedError
        
class LinearModelSimple(LinearModel):

    """
        Binary Class: 
        
        This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task.
        The predicted class corresponds to the sign of the regressor’s prediction
       
        Multi-classes: 

        Multi-output regression in the multiclass case.
        For multiclass classification, the problem is treated as multi-output regression, 
        and the predicted class corresponds to the output with the highest value.
    Args:
        LinearModel (_type_): _description_
    """

    def __init__(self, n_features, *, coefficients, intercepts, n_classes=None, learner_information=None, feature_names=None, class_names=(0, 1)):
        super().__init__(n_features, coefficients=coefficients, intercepts=intercepts, n_classes=n_classes, learner_information=learner_information, feature_names=feature_names, class_names=class_names)

    
    def _to_cxx(self):
        return (self.class_names, self.coefficients, self.intercepts)
                
    def _predict_instance(self, instance):
        """
        Return the prediction (the classification) of an observation (instance) according to the model
        """
        if len(self.class_names) <= 2:
            # Binary class: the predicted class corresponds to the sign of the regressor’s prediction.
            # prediction_raw_model = self.learner_information.raw_model.predict(instance.reshape(1, -1))
            regression_value = self.intercepts[0] + sum(instance[i]*self.coefficients[0][i] for i in range(len(instance)))
            return 1 if regression_value >0 else 0
            
        else:
            # Multic class: Multi-output regression in the multiclass case. 
            # For multiclass classification, the problem is treated as multi-output regression, 
            # and the predicted class corresponds to the output with the highest value.
            # prediction_raw_model = self.learner_information.raw_model.predict(instance.reshape(1, -1))
            
            regression_values = [self.intercepts[i] + sum(instance[j]*self.coefficients[i][j] for j in range(len(instance))) for i in range(len(self.class_names))]
            return self.class_names[numpy.argmax(regression_values)]


class LinearModelLogistic(LinearModel):
    """
        Despite its name (LogisticRegression in sklearn), it is implemented as a linear model for classification rather than regression in terms of the scikit-learn/ML nomenclature
        
        Logistic regression is a special case of Generalized Linear Models with a Binomial / Bernoulli conditional distribution and a Logit link. 
        The numerical output of the logistic regression, which is the predicted probability, can be used as a classifier by applying a threshold (by default 0.5) to it. This is how it is implemented in scikit-learn, so it expects a categorical target, making the Logistic Regression a classifier.

        Binary Class: 
        
        This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task.
        The predicted class corresponds to the sign of the regressor’s prediction
       
        Multi-classes: 

        Multi-output regression in the multiclass case.
        For multiclass classification, the problem is treated as multi-output regression, 
        and the predicted class corresponds to the output with the highest value.
    Args:
        LinearModel (_type_): _description_
    """

    def __init__(self, n_features, *, coefficients, intercepts, n_classes=None, learner_information=None, feature_names=None, class_names=None):
        super().__init__(n_features, coefficients=coefficients, intercepts=intercepts, n_classes=n_classes, learner_information=learner_information, feature_names=feature_names, class_names=class_names)
    
    def _to_cxx(self):
        return (self.class_names, self.coefficients, self.intercepts)
    
    def _predict_instance(self, instance):
        """
        Return the prediction (the classification) of an observation (instance) according to the model
        """
        if len(self.class_names) <= 2:
            # Binary class: the predicted class corresponds to the sign of the regressor’s prediction.
            # prediction_raw_model = self.learner_information.raw_model.predict(instance.reshape(1, -1))
            
            regression_value = self.intercepts[0] + sum(instance[i]*self.coefficients[0][i] for i in range(len(instance)))
            return 1 if regression_value >0 else 0
            
        else:
            # Multic class: Multi-output regression in the multiclass case. 
            # For multiclass classification, the problem is treated as multi-output regression, 
            # and the predicted class corresponds to the output with the highest value.
            # prediction_raw_model = self.learner_information.raw_model.predict(instance.reshape(1, -1))
            
            regression_values = [self.intercepts[i] + sum(instance[j]*self.coefficients[i][j] for j in range(len(instance))) for i in range(len(self.class_names))]
            return self.class_names[numpy.argmax(regression_values)]
        #raise NotImplementedError