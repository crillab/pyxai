import c_explainer
import time

from pyxai.sources.explainers.Explainer import Explainer
from pyxai.sources.tools.utils import count_dimensions
from pyxai.sources.data_structures.linearModel import LinearModel, LinearModelLogistic, LinearModelSimple


class ExplainerLinear(Explainer):

    def __init__(self, linear_model, instance=None):
        super().__init__()
        self._linear_model = linear_model
        self.c_linear_model = None
        self.upper_bounds = None
        self.lower_bounds = None
        
        if instance is not None:
            self.set_instance(instance)

    def set_upper_lower_bounds(self, instances):
        if self._instance is None:
            raise ValueError("Please call set_instance() before.")
        
        self.upper_bounds = [float("-inf")]*len(self._instance)
        self.lower_bounds = [float("inf")]*len(self._instance)

        for instance in instances: 
            if len(instance) != len(self._instance):
                raise ValueError("Each instance have to have the same size of the instance: "+str(instance) + " != " + str(self._instance))
            for i, value in enumerate(instance):
                if value > self.upper_bounds[i]:
                    self.upper_bounds[i] = value
                if value < self.lower_bounds[i]:
                    self.lower_bounds[i] = value


    def set_instance(self, instance):
        if instance is None:
            raise ValueError("The instance parameter is None.")

        if count_dimensions(instance) != 1:
            raise ValueError("The instance parameter should be an iterable of only one dimension (not " + str(
                count_dimensions(instance)) + ").")

        # The target observation.
        self._instance = instance
        # No binary representation in linear models 
        self._binary_representation = None 

        # The target prediction
        self.target_prediction = self.predict(self._instance)
        self.set_excluded_features(self._excluded_features)

    @property
    def linear_model(self):
        return self._linear_model

    def predict(self, instance):
        return self._linear_model.predict_instance(instance)
    
    def model_specific_reason(self, n_iterations=50, time_limit=None, seed=0, history=True):
        print("model_specific_reason")
        
        if self._instance is None:
            raise ValueError("Please call set_instance() before.")
        if self.upper_bounds is None or self.lower_bounds is None:
            raise ValueError("Please call set_upper_lower_bounds() before.")
        
        if seed is None:
            seed = -1
        if time_limit is None:
            time_limit = 0

        print("self.upper_bounds:", self.upper_bounds)
        print("self.lower_bounds:", self.lower_bounds)

        if self.c_linear_model is None:
            # Preprocessing to give the model in the c++ library
            print("self._linear_model.class_names:", self._linear_model.class_names)
            print("self._linear_model.coefficients:", self._linear_model.coefficients)
            print("self._linear_model.intercepts:", self._linear_model.intercepts)
            self.c_linear_model = c_explainer.new_linear_model(*self._linear_model._to_cxx())
        
        reason = c_explainer.model_specific_reason(self.c_linear_model, tuple(self._instance), tuple(self.upper_bounds), tuple(self.lower_bounds), n_iterations, time_limit, "none")
  
        feature_names = self.get_feature_names()

        str_reason = []
        for i, value in enumerate(reason):
            if value == True:
                str_reason.append(str(feature_names[i]) +" = "+ str(self._instance[i]))
        return tuple(str_reason)
        #Compute the reason
