from pyxai import Builder, Learning, Explainer, Tools
import unittest

Tools.set_verbose(0)


class TestMisc(unittest.TestCase):
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
            cls.model = cls.learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
        return cls.learner, cls.model


    def no_instance(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        try:
            explainer.direct_reason()
        except ValueError:
            raise


    def test_noinstance(self):
        with self.assertRaises(ValueError):
            self.no_instance()


if __name__ == '__main__':
    unittest.main(verbosity=2)
