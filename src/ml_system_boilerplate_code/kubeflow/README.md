# MACHINE LEARNING SYSTEM BOILERPLATE TEMPLATE

## Logic Diagram
![mle_pipeline](https://github.com/JosephWoodall/turbo-barnacle/blob/main/src/ml_system_boilerplate_code/kubeflow/assets/ML_SYSTEM_BOILERPLATE_CODE_DIAGRAM_V1.jpg?raw=true)

## Architecture Diagram
![mle_pipeline](https://github.com/JosephWoodall/turbo-barnacle/blob/main/src/ml_system_boilerplate_code/kubeflow/assets/ML_SYSTEM_BOILERPLATE_CODE_ARCHITECTURE_V1?raw=true)


### TECH STACK

Colons can be used to align columns.

| Tool| Purpose / Description|Technical Capability|MLOps Process Enabled|
|:-----:|:-----:|:-----:|:-----:|
| Kubeflow      | Managing end-to-end ML workflows on Kubernetes | ML Pipeline Development |- ML Development <br/> - Training Operationalization|
| MLFlow      | ML lifecycle management (tracking experiments, metadata, and artifacts) | Experiment & Metadata Tracking |- Model Deployment <br/> - Model Management|
| Seldon      | Deploying Machine Learning models at scale on Kubernetes | - Model Deployment <br/> - Model Serving |- ML Deployment <br/> - Prediction Serving|
| Prometheus      | Monitoring metrics collection and alerting toolkit | Monitoring | Continuous Monitoring |
| Grafana      | Dashboards to query, visualize, and alert on monitoring metrics | Monitoring | Continuous Monitoring |
| GoCD      | Automating training the deployment pipelines | Automation | Continuous Training |
| Streamlit      | Creating user interface to interact with ML models | Model Serving | Prediction Serving (Consumption) |
