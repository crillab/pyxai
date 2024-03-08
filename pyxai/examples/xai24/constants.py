from pyxai import  Learning
training_size = 0.7
classified_size = 0.5  # ratio with remaining instances
theta = 0.5
N = 50
debug = True
trace = True

USER_BT = 0
USER_LAMBDA = 1
user = USER_LAMBDA

model = Learning.RF
n_iterations = 50
max_time = 3600*3  # 4 hours
statistics = {"rectifications": 0, "generalisations": 0, "cases_1": 0, "cases_2": 0, "cases_3": 0, "cases_4": 0,
              "cases_5": 0, "n_positives": 0, "n_negatives": 0, "rectifications_times": [], "rectifications_cases":[]}
