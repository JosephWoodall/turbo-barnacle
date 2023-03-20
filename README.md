# turbo-barnacle
Reusable library! Hooray!

### Accelerator Directory Tree
Below is a nice representation of a reusable library (similar to turbo-barnacle, but more refined) ((in fact, you could even use some of turbo-barnacle's code within the below files! wow!)) just in case you would want to help accelerate some code development for any data science/data engineering related project. 

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
