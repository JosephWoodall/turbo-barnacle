# MACHINE LEARNING SYSTEM BOILERPLATE TEMPLATE

![mle_pipeline](https://github.com/JosephWoodall/turbo-barnacle/blob/main/src/data_science_accelerator_example/python/assets/ml-engineering.jpg?raw=true)
source: https://ml-ops.org/content/end-to-end-ml-workflow

## OVERVIEW
One of the core tenants of data science that differentiates it from software engineering is its focus on experimentation. In software engineering, you develop, test, and push features that are primarily code-based. In data science, you conduct heaps of experiments while making changes in configuration, data, features, etc...The output isn't even necessarily "completed code," but artifacts such as model weights. Furthermore, there are different components of an ML system. "ML algorithms" is usually what people think of when they say machine learning, but it's only a small part of the entire system.

Please also keep in mind, for any ML project to succeed within a business organization, it's crucial to tie the performance of an ML system to the overall business performance. The effect of an ML project on business objectives can be hard to reason about. To gain a definite answer on the question of how ML metrics influence business metrics, experiments like A/B testing are often needed, regardless of model performance. Returns on investment in ML depend a lot on the maturity stage of adoption. The longer an organization has adopted ML, the more efficient your pipeline will run, the faster your development cycle will be, and the less engineering time you'll need, as well as the lower yourcloud bills will be, all of which lead to higher returns. According to Algorithmia, among companies that are more sophisticated in their ML adoption (having had models in production for over 5 years), almost 75% can deploy a model in under 30 days. Among those just getting started with their ML pipeline, 60% take over 30 days to deploy a model.

### ML SYSTEM CHARACTERISTICS
Most ML systems should have the following characteristics:  
- Reliability: the system should continue to perform the correct function at the desired level of performance even in the face of adversity (hardware or software faults, and even human error).  
- Scalability: the system should have reasonable ways to up-scale or down-scale depending on usage.  
- Maintainability: it's important to structure workloads and set up infrastructure in a way that different contributors can work using tools they're comfortable with, instead of one group of contributors forcing their tools on other groups. Code should be documented. Code, data, and artifacts should be versioned. Models should be sufficiently reproducible so that even when the original authors are not around, other contributors can have sufficient contexts to build on their work.  
- Adaptability: the system should have some capacity for both discovering aspects for performance improvement and allowing updates without service interruption. Because ML systems are part code, part data, and data can change quickly, ML systems need to be able to evolve quickly. This is tightly linked to maintainability.  

### USUAL (SIMPLIFIED) STEPS INVOLVED IN DEVELOPING ML SYSTEMS
Step 1: Project Scoping:  
- A project starts with scoping the project, laying out goals, objectives, and constraints. Stakeholders should be identified and involved. Resources should be estimated and allocated.  

Step 2: Data Engineering:  
- A vast majority of ML models today learn from data, so developing ML models starts with engineering data.  

Step 3: ML Model Development:   
- With the initial set of training data, we'll need to extract features and develop initial models leveraging these features. This is the stage that requires the most ML knowledger and is most often covered in ML courses.  

Step 4: Deployment:  
- After a model is developed, it needs to be made accessible to users. Developing an ML system is like writing- you'll never reach the point when your system is done. But you do reach the point when you have to put your system out there.  

Step 5: Monitoring and Continual Learning:  
- Once in production, models need to be monitored for performance decay and maintained to be adaptive to changing environments and changing requirements.  

Step 6: Business Analysis:  
- Model performance needs to be evaluated against business goals and analyzed to generate business insights. These insights can then be used to eliminate unproductive projects or scope out new projects. This step is closely related to the first step.  

### DS HIERARCHY OF NEEDS (CANNOT HAVE ONE WITHOUT THE PREVIOUS ONES)
- Collect: Instrumentation, logging, sensors, external data (second or third-party data), internal data (first-party data), user-generated content.  
- Move/Store: Reliable data flow, infrastructure pipelines, ETL/ELT, structured and unstructured data storage.  
- Explore/Transform: Cleaning, anomaly detection, prep.  
- Aggregate/label: analytics, metrics, segments, aggregattes, features, training data.  
- Learn/Optimize: A/B testing, experimentation, simple ML algorithms.  
- AI, Deep Learning.  

### PROBLEM STATMENT
[Use this space to fill in the exact problem statment you wish to solve here.  
Clearly note and describe:  
- what the input is  
- what the output is  
- what is the objective function  
- what kind of task is it  
]

### GATHER AND CLEAN DATA
[Use this space as a data dictionary for data relevant to the problem you want to solve, and clean/preprocess the data to ensure it's ready for modeling.  
- What is/are the data source(s)  
- What is/are the data source(s) formats? JSON, CSV, Parquet, Avro, Protobuf, Pickle?  
- What is the data model? Relational? NoSQL? Graph? Structured or Unstructured?
- Desribe the ETL/ELT process, if any.  
- What is the dataflow mode? Data passing through databases? RESTful (POST/GET)? RPC? Real-time via Kafka/Kenesis? PubSub/Message Queue? Batch or Stream?  
]

### EXPLORE AND VISUALIZE DATA
[Use this space to make note of any interesting findings, patterns, or relationships from the EDA that might inform your modeling choices.  
- Did your findings come from random sampling/non probability sampling/stratified sampling/weighted sampling/reservoir sampling/importance sampling?  
- Are you going to be using supervised or unsupervised ML? If supervised, did your data come with hand-labels/natural labels or did you create them?
]

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