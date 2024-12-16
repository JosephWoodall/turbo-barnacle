import time 

import pandas 
import numpy 
from contextlib import redirect_stdout

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score

import xgboost 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.feature_selection import SelectFromModel

def load_production_data():
    pass 

def load_generated_data():
    num_rows = 100000
    generated_data = pandas.DataFrame({
        'test_feature_1':pandas.Categorical(numpy.random.choice(['A', 'B', 'C'], num_rows)),
        'test_feature_2':numpy.random.randint(0, 100, size = num_rows),
        'test_feature_3':pandas.Categorical(numpy.random.choice(['A', 'B', 'C'], num_rows)),
        'test_feature_4':numpy.random.randint(0, 100, size = num_rows),
        'test_response_binary':numpy.random.choice([0, 1], num_rows)
        })
    print(generated_data)
    return generated_data 

def eda(focus_data: pandas.DataFrame, filename):
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            print("-"*100)
            print(f"{filename}")
            print("-"*100)
            print("-"*25)
            print("Info")
            print("-"*25)
            print(focus_data.info())
            print("-"*25)
            print("Describe")
            print("-"*25)
            print(focus_data.describe())
            print("-"*25)
            print("Nulls per Feature")
            print("-"*25)
            total_nulls = focus_data.isnull().sum(axis = 0)
            percent_nulls = (focus_data.isnull().sum() / len(focus_data)) * 100
            null_summary = pandas.DataFrame({
                'total_null_values':total_nulls,
                'percent_of_null_values':percent_nulls
                })
            print(null_summary)
            print("-"*25)
            print("Features with more than 10% null values")
            print("-"*25)
            high_null_features = null_summary[null_summary['percent_of_null_values'] > 10]
            print(high_null_features)
            
if __name__ == '__main__':
    data = load_generated_data()
    eda(data, "eda.txt")

