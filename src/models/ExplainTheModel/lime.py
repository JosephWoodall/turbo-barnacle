import lime 
import lime.lime_tabular
import numpy as np 

model = '' # load the model here
x_train = '' # load the training data here
x_test = '' # load the test data that you want to explain here
feature_names = '' 
class_names = '' 

explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names = feature_names, class_names = class_names, kernel_width = 3)

i = '' # choose an index of the test instance that you want to explain 
x = x_test[i]

exp = explainer.explain_instance(x, model.predict_proba, num_features = 5, top_labels =  1)

# prints the explanation
print(exp.as_list())