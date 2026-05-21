# Check V1.0: Ok
# Check V2.0: Ok

from pyxai import Learning, Explaining, Tools

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, problem_type='regression', instances_type='tabular')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.LINEAR_RIDGE, splitting_parameters={'test_size':0.2}, model_parameters={'alpha':0.5})
instance, prediction = learner.get_instances(model, n=1)

print("instance:", instance)
print("prediction:", prediction)
