class LearnerInformation:
    def __init__(self, raw_model, training_index=None, test_index=None, group=None, metrics=None, extras=None):
        self.raw_model = raw_model
        self.training_index = training_index
        self.test_index = test_index
        self.group = group
        self.metrics = metrics
        self.extras = extras
        self.learner_name = None
        self.feature_names = None
        self.evaluation_method = None
        self.evaluation_output = None

    def to_dict(self):
        learner_dict = {}
        learner_dict["raw_model"] = self.raw_model
        learner_dict["training_index"] = self.training_index
        learner_dict["test_index"] = self.test_index
        learner_dict["group"] = self.group
        learner_dict["metrics"] = self.metrics
        learner_dict["extras"] = self.extras
        learner_dict["learner_name"] = self.learner_name
        learner_dict["feature_names"] = self.feature_names
        learner_dict["evaluation_method"] = self.evaluation_method
        learner_dict["evaluation_output"] = self.evaluation_output
        return learner_dict
        
            

    def set_learner_name(self, learner_name):
        self.learner_name = learner_name


    def set_feature_names(self, feature_names):
        self.feature_names = feature_names


    def set_evaluation_method(self, evaluation_method):
        self.evaluation_method = str(evaluation_method)


    def set_evaluation_output(self, evaluation_output):
        self.evaluation_output = str(evaluation_output)