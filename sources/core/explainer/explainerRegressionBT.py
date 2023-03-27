import c_explainer

class ExplainerRegressionBT(ExplainerBT) :
    def __init__(self, boosted_trees, instance=None):
        super().__init__(boosted_trees, instance)

    