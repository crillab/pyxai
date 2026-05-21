import Learning
import pandas as pd
import datetime

df = pd.read_csv('tests/weatherAUS.csv')
month_lst = [datetime.datetime.strptime(x, '%Y-%m-%d').month for x in df.Date]
df.Date = month_lst
df = df.rename(columns={'Date':'Month'})


preprocessor = Learning.TabularPreprocessor(df, target_feature="RainTomorrow", problem_type=Learning.CLASSIFICATION, classification_type='binary-class') 
preprocessor.set_categorical_features(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])
preprocessor.set_default_type(Learning.TypeFeature.Numerical)

preprocessor.process() 
preprocessor.export("weatherAUS_encoded.csv")