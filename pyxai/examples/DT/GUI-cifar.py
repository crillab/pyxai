from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv
# Check V1.0: Ok
import numpy

# Dataset
dataset = "./examples/datasets_not_converted/cifar_cat_dog.csv"

# Tuning
def tuning():
    import pandas
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    def load_dataset(dataset):
        data = pandas.read_csv(dataset).copy()

        # extract labels
        labels = data[data.columns[-1]]
        labels = numpy.array(labels)

        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])

        # extract the feature names
        feature_names = list(data.columns)

        return data.values, labels, feature_names

    X, Y, names = load_dataset(dataset)
    model1 = RandomForestClassifier()
    param_grid = {'n_estimators': [100, 250, 500], 'max_depth': [5, 8, 12], 'max_features' : [5, 10, 16] }
    gridsearch1 = GridSearchCV(model1, 
                            param_grid=param_grid, 
                            scoring='balanced_accuracy', refit=True, cv=3, 
                            return_train_score=True, verbose=10)

    gridsearch1.fit(X, Y)
    return gridsearch1.best_params_
best_parameters = tuning()
print("Best parameters from tuning:", best_parameters)

# Machine learning part
learner = Learning.Scikitlearn(dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, **best_parameters)
instances = learner.get_instances(model, n=10, correct=True)

explainer = Explainer.initialize(model)
    
for (instance, prediction) in instances:
    explainer.set_instance(instance)
    sufficient = explainer.sufficient_reason()
    print("suffisante done")
    minimals = explainer.minimal_majoritary_reason(n=1, time_limit=60)
    print("minimal done")
    
#Get from the position of a pixel 'x' and 'y' a color value (r, g, b) according to an 'instance'

def get_pixel_value(instance, x, y, shape):
    n_pixels = shape[0]*shape[1]
    index = x * shape[0] + y 
    return (instance[0:n_pixels][index], instance[n_pixels:n_pixels*2][index],instance[n_pixels*2:][index])

def instance_index_to_pixel_position(i, shape):
    n_pixels = shape[0]*shape[1]
    if i < n_pixels:
        value = i 
    elif i >= n_pixels and i < n_pixels*2:
        value = i - n_pixels  
    else:
        value = i - (n_pixels*2)  
    return value // shape[0], value % shape[0]
        
explainer.show(image={"shape": (32,32,3),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})
