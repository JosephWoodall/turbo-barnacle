from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier

"""
The script defines a FeatureSelector class that takes in a data set and target variable as input. 
The select_kbest method selects the top k features based on the univariate statistical test (f_classif) and 
the select_model_based method selects the features by training an ExtraTreesClassifier on the data and returning the feature importance. 
Then the get_final_features method takes the selected features and returns the final list of explanatory features with the response variable.

You can use either of these methods as per your requirement or even combine the features obtained from both the methods.

It's important to note that feature selection is an iterative process and the feature set that is selected may change as you experiment 
with different models and techniques
"""

class FeatureSelector:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.features = None
        
    def select_kbest(self, k=10):
        selector = SelectKBest(f_classif, k=k)
        selector.fit(self.data, self.target)
        self.features = selector.get_support(indices=True)
        return self.features
    
    def select_model_based(self):
        model = ExtraTreesClassifier()
        model.fit(self.data, self.target)
        self.features = model.feature_importances_
        return self.features
   
    def get_final_features(self):
        data_cols = list(self.data.columns)
        final_features = [data_cols[i] for i in self.features]
        final_features.append("target_column")
        return final_features