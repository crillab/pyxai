# Check V1.0: Ok
# Check V2.0: Not Ok

from pyxai import Learning, Explaining, Tools


# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, problem_type='regression', instances_type='tabular', labels_type='classes')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.LINEAR_SIMPLE, splitting_parameters={'test_size':0.2})
instance, prediction = learner.get_instances(model, n=1)

print("instance:", instance)
print("prediction:", prediction)
