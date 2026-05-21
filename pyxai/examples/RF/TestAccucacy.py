from pyxai import Learning, Tools, Explaining

import statistics
import numpy

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, problem_type=Learning.CLASSIFICATION)
models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.RF, splitting_parameters={"n_models": 10})


def extract_mean_std(l):
    m = statistics.mean(l)
    s = numpy.std(l)
    return str(format(round(m, 3), '.3f')) + " ("+str(format(round(s, 3), '.3f'))+")"

accuracies = []
n_variables = []
n_booleans = []
for i, model in enumerate(models):
    accuracies.append(learner.learner_information_lst[i].metrics["accuracy"])
    n_nodes = 0
    for tree in model.forest:
        n_nodes += tree.n_nodes()
    n_variables.append(n_nodes)
    n_booleans.append(len(model.map_features_to_id_binaries.keys()))

#print("accuracies:", extract_mean_std(accuracies))
#print("n_variables:", extract_mean_std(n_variables))
#print("n_booleans:", extract_mean_std(n_booleans))
print(extract_mean_std(accuracies) + " & " + extract_mean_std(n_variables) + " & " + extract_mean_std(n_booleans))