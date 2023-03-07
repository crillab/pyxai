# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime
import pandas

data = pandas.read_pickle("../../list_interpolated_series.p")

print("data", data)

