from pyxai import Learning

learner = Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.DT)
for id, models in enumerate(models):
     metrics = learner.get_details()[id]["metrics"]