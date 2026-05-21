from pyxai import Learning, Explaining, Tools
import pickle
import os
# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, problem_type=Learning.CLASSIFICATION)
models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.RF, splitting_parameters={"n_models":10})

name = Tools.Options.dataset.split("/")[-1].split(".")[0]

for i, model in enumerate(models):
    print("model", i, "export:")
    raw_model = learner.learner_information_lst[i].raw_model
    filename = Tools.Options.output+os.sep+name+"_model_"+str(i)
    file = open(filename + ".pkl", 'wb')
    pickle.dump(raw_model, file)
    file.close()
    instances = learner.get_instances(model, indexes=Learning.TEST, n=10, details=True)
    file = open(filename + ".instances", 'w')
    for instance in instances:
        #print("instance:", list(instance["instance"]))
        #exit(0)
        file.write(" ".join([str(x).strip() for x in instance["instance"]]) + "\n") 
    file.close()