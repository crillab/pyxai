import constants
from pyxai import Learning, Explainer

class Model:
    def __init__(self, learner_AI):
        self.learner = learner_AI
        self.model = learner_AI.evaluate(method=Learning.HOLD_OUT, output=constants.model, test_size=1 - constants.training_size, seed=123, min_samples_split=0.5)
        self.explainer = None
    def set_instance(self, instance):
        self.explainer.set_instance(instance)


    def reason(self, *, n_iterations=None):
        if constants.model == Learning.RF:
            if n_iterations is None:
                n_iterations = constants.n_iterations
            return self.explainer.majoritary_reason(n_iterations=n_iterations, seed=123)
        if constants.model == Learning.DT:
            reason = self.explainer.sufficient_reason()
            return reason
        return None

    def predict_instance(self, instance):
        return self.model.predict_instance(instance)

