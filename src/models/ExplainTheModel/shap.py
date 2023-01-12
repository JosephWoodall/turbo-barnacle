import shap 
import matplotlib.pyplot as plt 

model = '' # load the model here
x_train = '' # load the training data here

explainer = shap.TreeExplainer(model)

# explain the model's predictions on the training data 
shap_values = explainer.shap_values(x_train)

for i in range(x_train.shape[1]):
    plt.subplot(x_train.shape[1], 1, i+1)
    shap.summary_plot(shap_values[:, i], x_train, plot_type = 'bar', show = False)
    plt.show()