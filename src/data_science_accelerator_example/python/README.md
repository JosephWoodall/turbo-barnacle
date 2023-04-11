# END TO END MACHINE LEARNING ENGINEERING WORKFLOW BOILERPLATE TEMPLATE

![mle_pipeline](src/data_science_accelerator_example/python/assets/ml-engineering.jpg)

## OVERVIEW

### PROBLEM STATMENT
[Use this space to fill in the exact problem statment you wish to solve here]

### GATHER AND CLEAN DATA
[Use this space as a data dictionary for data relevant to the problem you want to solve, and clean/preprocess the data to ensure it's ready for modeling]

### EXPLORE AND VISUALIZE DATA
[Use this space to make note of any interesting findings, patterns, or relationships from the EDA that might inform your modeling choices]

### DEVELOP AND TRAIN MODELS
[Use this space to explain the model of choice, or type of model of choice, and explain any evaluation methodologies for selecting the best model here]

### DEPLOY MODELS
[Use this space to explain where the best model will be deployed. Will it be a web service/api call? Will it be a batch prediction model living in the backend? Will it be an embedded model in an edge device? List this here]

### MONITOR AND MAINTAIN MODELS
[Use this space to explain how the deployed model's performance is being monitored over time and how it will be re-trained as necessary to ensure it continues to produce accurate/quality predictions]

### SCALE UP AND OPTIMIZE
[Use this space to explain how the deployed model's scalability is accounted for if/when its usage grows. Will infrastructure scale up? Will you optimize the model's deployment to handle increased traffic? List this here]

```bash
data_science_accelerator/
├── README.md
├── Dockerfile
├── data_prep/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── transformation.py
│   ├── feature_engineering.py
│   ├── validation.py
│   ├── retrieval.py
│   ├── storage.py
│   └── pipeline.py
├── data_viz/
│   ├── __init__.py
│   ├── plot_utils.py
│   └── dashboard.py
├── model_training/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_selection.py
│   ├── model_evaluation.py
│   └── model_export.py
├── model_deployment/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── serving.py
│   └── monitoring.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   └── metrics.py
├── tests/
│   ├── __init__.py
│   ├── test_data_prep.py
│   ├── test_data_viz.py
│   ├── test_model_training.py
│   ├── test_model_deployment.py
│   └── test_utils.py
└── examples/
    ├── __init__.py
    ├── data_prep_example.ipynb
    ├── data_viz_example.ipynb
    ├── model_training_example.ipynb
    ├── model_deployment_example.ipynb
    └── utils_example.ipynb
```