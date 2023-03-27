from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.LightGBM(Tools.Options.dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, learner_type=Learning.REGRESSION, 
                         n_estimators=100)
instances = learner.get_instances(model=model, n=2)

instance = instances[1][0]
prediction = instances[1][1]

# Explanation part
print("instance:", instance)
print("prediction classifier:", format(prediction, 'f'))
print("prediction model:", model.predict_instance(instance))
#explainer = Explainer.initialize(model, instance)
#direct_reason = explainer.direct_reason()
#print("len direct: ", len(direct_reason))
#print("is a reason (for 50 checks):", explainer.is_reason(direct_reason, n_samples=50))


