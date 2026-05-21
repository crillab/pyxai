from pyxai import Builder, Learning, Explaining, Tools
import unittest

Tools.set_verbose(0)


class TestExplainingMisc(unittest.TestCase):
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/datasets/iris.csv", problem_type='classification', instances_type='tabular', labels_type='classes')
            cls.model = cls.learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.DT, model_parameters={})
        return cls.learner, cls.model


    def no_instance(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        try:
            explainer.direct_reason()
        except ValueError:
            raise


    def test_noinstance(self):
        with self.assertRaises(ValueError):
            self.no_instance()


    def test_format(self):
        self.assertTrue(Explaining.Explainer.format([1, 2], n=1) == (1, 2))
        self.assertTrue(Explaining.Explainer.format([[1, 2], [3]], n=2) == ((1, 2), (3,)))
        self.assertTrue(Explaining.Explainer.format([[1]], n=2) == ((1,),))


if __name__ == '__main__':
    unittest.main(verbosity=1)
