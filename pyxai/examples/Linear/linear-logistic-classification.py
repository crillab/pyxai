# Check V1.0: Ok
# Check V2.0: Ok but: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names

from pyxai import Learning, Explaining, Tools

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, problem_type='classification')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.LINEAR_LOGISTIC, splitting_parameters={'test_size':0.2})
instance, prediction = learner.get_instances(model, n=1)

print("instance:", instance)
print("prediction:", prediction)
