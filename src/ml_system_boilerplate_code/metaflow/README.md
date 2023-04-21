# MACHINE LEARNING SYSTEM BOILERPLATE TEMPLATE

![mle_pipeline](https://github.com/JosephWoodall/turbo-barnacle/blob/main/src/ml_system_boilerplate_code/metaflow/assets/ML_SYSTEM_BOILERPLATE_CODE_DIAGRAM_V1.jpg?raw=true)

Fantastic Resources for ML Systems:  
- https://github.com/chiphuyen/dmls-book  
- https://ml-ops.org/content/end-to-end-ml-workflow  

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

## ML SYSTEM CHECKLIST
Developing an ML system from scratch is often a complex, daunting task because no one system is usually alike, and consist of many different components (ML algorithms, data, business logic, evaluation metrics, underlying infrastructure, etc...) and involve many different stakeholders (data scientists, ML engineers, business leaders, users, even society at large sometimes). ML systems are unique because they are data dependent, and data varies widly from one use case to the next.  

As such, the below checklist will at least serve as a template to have when you start these conversations. Please fill out each part, dilligently, as it might just save you months worth of work later on.  

### PROBLEM STATMENT
Use this space to fill in the exact problem statment you wish to solve here.  
Clearly note and describe:  
- What the input is  
- What the output is  
- What is the objective function  
- What kind of task is it  


### GATHER AND CLEAN DATA
Use this space as a data dictionary for data relevant to the problem you want to solve, and clean/preprocess the data to ensure it's ready for modeling.  
Clearly note and describe:    
- What is/are the data source(s)  
- What is/are the data source(s) formats? JSON, CSV, Parquet, Avro, Protobuf, Pickle?  
- What is the data model? Relational? NoSQL? Graph? Structured or Unstructured?
- Desribe the ETL/ELT process, if any.  
- What is the dataflow mode? Data passing through databases? RESTful (POST/GET)? RPC? Real-time via Kafka/Kenesis? PubSub/Message Queue? Batch or Stream?  


### EXPLORE AND VISUALIZE DATA
Use this space to make note of any interesting findings, patterns, or relationships from the EDA that might inform your modeling choices.  
#### TRAINING DATA
Clearly note and describe:  
- Did your findings come from random sampling/non probability sampling/stratified sampling/weighted sampling/reservoir sampling/importance sampling?  
- Are you going to be using supervised or unsupervised ML? If supervised, did your data come with hand-labels/natural labels or did you create them?  
- Was there class imbalance that needed to be corrected? If so, how did you correct it? Data-level methods/Algorithm-level methods?  
- Did you resample data-levels? If so, how did you resample? Oversample/Undersample?  
- Did you augment the data? If so, which technique did you use? Simple Label-Preserving Transformations/Pertubation/Data Synthesis
#### FEATURE ENGINEERING
Clearly note and describe:  
- Were there missing data values? If so, were they missing not at random, missing at random, or missing completely at random? How did you correct this?  Imputation/Deletion? If Imputation, which methods did you use to impute? If Deletion, why did you delete the records with missing values?  
- Did you scale/normalize the continuous features? Did you discretize the features? Did you encode the categorical features? Did you feature cross? Did you include discrete/continuous positional embeddings?  
- Was there Data Leakage? Was there group leakage? What was the cause? Did you split your time-correlated data by time instead? How did you detect the data leakage?  
- Which features did you engineer? (see below for rules of thumb):  
    - The more features you have, the more opportunities there are for data leakage.  
    - Too many features can cause overfitting.  
    - Too many features can increase memory required to serve a model, which, in turn, might require you to use more expensive machine/instance to serve your model.  
    - Useless features become technical debts. Whenever your data pipeline changes, all the affected features need to be adjusted accordingly. For example, if one day your application decides to no longer take in information about users' age, all features that use users' age need to be updated.  
- Did you include functionality for feature selection/importance? If not, you should. If so, which technique(s) did you use to select those features?  
- Rules of thumb:  
    - Split data by time into train/test/validation splits instead of doing it randomly.  
    - If you oversample your data, do it after splitting.  
    - Scale and normalize your data after splitting to avoid data leakage.  
    - Use statistics from only the train split, instead of the entire data, to scale your features and handle missing values.  
    - Understand how your data is generated, collected, and processed. Involve domain experts if possible.  
    - Keep track of your data's lineage.  
    - Understand feature importance to your model.  
    - Use features that generalize well.  
    - Remove no longer useful features from your models.  

### DEVELOP AND TRAIN MODELS
Use this space to explain the model of choice, or type of model of choice, and explain any evaluation methodologies for selecting the best model here. 
- Rules of thumb:  
    - Avoid the state-of-the-art trap.  
    - Start with the simplest methods.  
        - Simpler models are easier to deploy, starting simple and adding to it makes it super easy to debug, simple models serve as a nice baseline to which to compare complex models, and Ensembled models are really, really good, homie.  
    - Avoid human bias in selecting models.  
    - Evaluate good performance now versus good performance later.  
    - Evaluate trade-offs.  
    - Understand your model's assumptions.  
        - "All models are wrong, but some are useful." -George Box, 1976.  
- Did you use an Ensemble/single model? Bagging? Boosting? Stacking?  
- Did you track your experiments and include the following:  
    - Loss Curve
    - Model Performance (like accuracy, F1, perplexity, recall, precision, etc...)  
    - Speed of Model  
    - System Performance Metrics (such as CPU/GPU utilization)  
    - Parameter/Hyperparameter changes over time? Gradient norms, weight norms, etc...  
- Did you version your experiments?  
- Did you have to do any debugging of your model?  
    - Debugging ML models is hard because models fail silently, it's slow to validate if you fixed the bug, and models are very cross-functionally complex.  
    - Common reasons an ML model failed:  
        - Theoretical Constraints  
        - Poor Implementation of Model  
        - Poor Choice of Hyperparameters  
        - Data Problems  
        - Poor Choice of Features  
    - Debugging Techniques:  
        - Start simple and gradually add more components.  
        - Overfit a single batch.  
        - Set a random seed.  
- Did you distribute the training? If so, how? Data parrallelism? Model parallelism?
- Did you use AutoML? If so, did you use soft (hyperparameter tuning optimization) or hard (architecture search and learned optimizer) AutoML? Why?  
    - If you used Hard AutoML, include a function to showcase the final architecture!  
- What were your offline evaluation metrics? Simple heuristics/zero rule baseline/human baseline/existing solutions?  
- What were your offline evaluation methods? Pertubation tests/invariance tests/discretional expectation tests/model calibration/confidence measurement/slice-based evaluation?  

### DEPLOY MODELS
Use this space to explain where the best model will be deployed. Will it be a web service/api call? Will it be a batch prediction model living in the backend? Will it be an embedded model in an edge device? List this here.  
- Who are the end users? Are you deploying to business users, who only need plots? Are you deploying and keeping your models up and running for millions of users a day?  
- Here are some myths on model deployment:  
    - You can only deploy one or two ML models at a time (you can actually deploy...a lot...)  
    - If we don't do anything, model performance remains the same (nope, models age like warm milk)  
    - You wont need to update your models as much (ask yourself, "how often am I reasonably able to update my models?")  
    - Most ML Engineers dont need to worry about Scale (um, yes they should)  
- Will you be deploying your model for Batch Prediction, Online Prediction, and/or to Edge Devices? What about a web browser?  
    - Batch prediction, which only uses batch features.  
    - Online prediction, which uses only batch features (e.g. precomputed embeddings).  
    - Online prediction, which uses both batch features and streaming features (also known as streaming prediction).  
    - Edge prediction, which is where computation is done on the edge on individual devices, where no internet connections are required, and network latency is not a concern.  
    - Browser prediction, which is an online prediction technique for running predictions in browser using WebAssembly.  
- How are you decreasing the latency for online predictions? Model Compression? Low-Rank Factorization? Knowledge Distillation? Pruning? Quantization?  
- How are you optimizing your model? Vectorization? Parallelization? Loop tiling? Operation Fusion? ML Techniques like autoTVM?

### MONITOR AND MAINTAIN MODELS
Use this space to explain how the deployed model's performance is being monitored over time and how it will be re-trained as necessary to ensure it continues to produce accurate/quality predictions. Remember, models age like warm milk.  
#### ML SYSTEM FAILURES
- Rules of Thumb for Failures:  
    - Software System Failures (Dependency, Deployment, Hardware, or Downtime/Crashing Failures)
    - ML Specific Failures (Production data differing from training data, Edge cases, Degenerate feedback loops (this is a biggie))
    - Data Distribution Shifts (Covariate Shift, Label Shift, Concept Drift)
#### MONITORING AND OBSERVABILITY
Monitoring refers to the act of tracking, measuring, and logging different metrics that can help determine when something goes wrong.  
Observability (also called instrumentation) refers to setting up the ML system in a way that gives us visibility into the system to help us investigate what went wrong, and is part of monitoring.  
- How are you monitoring the system?  
    - List your Operational Metrics here:  
    - List your ML-Specific Metrics here:  (accuracy-related metrics, predictions, features, and raw inputs)
- What are you including in your logs?  
- Are you using any Dashboards? What are they? Where are they?  
- What alerts do you have in place? What is your alert policy? What are the notification channels? What is your description format of your alerts?  

### CONTINUAL LEARNING AND TESTING IN PRODUCTION
Use this space to expalin how the deployed model will adapt to data distribution shifts in terms of your infrastructure. The goal of continual learning is to safely and efficiently automate the update of a model to adapt to the data, ensuring the ML system is maintainable and adaptable to changing environments, e.g., updating models that are in production in micro-batches, for example, updating the existing model after every 512 or 1,024 examples- the exact number is task dependent.  
- Will you use stateless retraining (training from scratch each time) or stateful training (model continues to train on new data)?  
- Does your system account for both types of retraining? If so, what is your training frequency for this deployed model? (hint hint: as often as you can, same goes for model updates, too)   
- How are you overcoming the challenges of continual learning? List your response to each of the below:  
    - Fresh Data Access:  
    - Evaluation:  
    - Algorithm:  
- What is your plan to test in production? Shadow deployment, A/B testing, Canary release, Interleaving experiments, Bandits?  

### INFRASTRUCTURE AND TOOLING FOR MLOPS SYSTEM
Use this space to summarize the infrastructure in use in order to account for EVERYTHING mentioned above in the previous sections. This is where you will list, in summary, each infrastructure component responsible for each component of the ML system. This will serve as a quick reference for stakeholders invested into this specific ML system. Infrastructure here refers to the set of fundamental facilities that support the development and maintenance of the ML system.  
Clearly outline and describe the usage of infrastructure components used in the following four, generalized layers (remember, multi-cloud is ok! and also remember, if you don't use Docker then I will be sad):  
- Storage and Compute:  
    - How much storage do you need? Which solution?  
    - How much compute do you need? Which solution? What is the GPU count, vCPU, memory, and GPU memory count?  
- Resource Management:  
    - Are you using Cron jobs?  
    - Are you using Schedulers?  
    - Are you using Orchestrators? (airflow, argo, prefect, kubeflow, metaflow, etc?...Metaflow is my preferred rm)  
- ML Platform:  
    - Model Deployment:  
    - Model Store (including model definition, model parameters, featurize/predict functions, dependencies, data, model generation code, experiment artifacts, tags)  
    - Feature Store (including feature management, feature computation, feature consistency)  
- Development Environment:  
    - What is/are your IDE(s)?  
    - How are you implementing Versioning?  
    - How are you implementing CI/CD?  
    - How are you standardizing the Dev Environment?  
    - Are you implementing container tech like Docker? Are you implementing Kubernetes to manage those instances?  
#### SCALE UP AND OPTIMIZE
Use this space to explain how the deployed model's scalability is accounted for if/when its usage grows. Will infrastructure scale up? Will you optimize the model's deployment to handle increased traffic?  
- Will you be using Docker? If so, how will you manage those containers? Kubernetes?  
- Will you be using a fully managed service to account for autoscaling? If so, which service?  

## MLOps Stack Canvas (CRISP-ML(Q))
For each ML Service created, please complete, in detail, answers to the following BEFORE development begins (preferably in a separate csv file stored under assets/):  

Project Name:   
Date:   
Team Members:   

0. 3 Dilemmas of MLOps:    
- Tooling: Should we buy, use existing open-source or build in-house tools for any of the MLOps components? What are the risks, trade-offs, and impacts of each of the decisions? 
- Platforms: Should we agree on one MLOps platform or create a hybrid solution? What are the risks, trade-offs, and impacts of each of the decisions? 
- Skills: How expensive is it to either acquire or educate our own machine learning engineering talents?  

1. Value Proposition:   
- What are we trying to do for the end-user(s)? 
- What is the problem?  
- Why is this an important problem? 
- Who is our persona? (ML Engineer, Data Scientist, Operation/Business User)    
- Who owns the models in production?    

2. Data Sources and Data Versioning:    
- Is this data versioning optional or mandatory? E.g., is data versioning a requirement for a system like a regulatory requirement? 
- What data sources are available? (e.g., owned, public, earned, paid data) 
- What is the storage for the above data? (e.g., data lake, DWH)    
- Is manual labeling required? Do we have human resources for it?   
- How to version data for each trained model?   
- What tooling is available for data pipelines/workflows?   

3. Data Analysis and Experiment Management
- What programming language to use for analysis? (R, Python, Scala, Julia. Or is SQL sufficient for analysis?)  
- Are there any infrastructure requirements for model training? 
- What ML-specific and business evaluation metrics need to be computed? 
- Reproducibility: What metadata about ML experiments is collected? (data sets, hyperparameters)  
- What ML Framework know-how is there?    

4. Feature Store and Workflows: 
- Is this optional or mandatory? Do we have a data governance process such that feature engineering has to be reproducible? 
- How are features computed (workflows) during the training and prediction phases?    
- What are infrastructure requirements for feature engineering?   
- “Buy or make” for feature stores?   
- What databases are involved in feature storage? 
- Do we design APIs for feature engineering?  

5. Foundations (Reflecting DevOps): 
- How do we maintain the code? What source version control system is used?    
- How do we monitor the system performance?   
- Do we need versioning for notebooks?    
- Is there a trunk-based development in place?    
- Deployment and testing automation: What is the CI/CD pipeline for the codebase? - - - What tools are used for it? 
- Do we track deployment frequency, lead time for changes, mean time to restore, and change failure rate metrics?   

6. Continuous Integration, Training, and Deployment: ML Pipeline Orchestration: 
- How often are models expected to be retrained? What is the trigger for it (scheduled, event-based, or ad hoc)?  
- Where does this happen (locally or on a cloud)? 
- What is the formalized workflow for an ML pipeline? (e.g., Data prep -> model training -> model eval & validation) What tech stack is used? 
- Is distributed model training required? Do we have an infrastructure for the distributed training?  
- What is the workflow for the CI pipeline? What tools are used?  
- What are the non-functional requirements for the ML model (efficiency, fairness, robustness, interpretability, etc.)? How are they tested? Are these tests integrated into the CI/CT workflow?  

7. Model Registry and Model Versioning: 
- Is this optional or mandatory? The model registry might be mandatory if you have multiple models in production and need to track them all. The reproducibility requirement might be the reason that you need the model versioning.  
- Where should new ML models be stored and tracked?   
- What versioning standards are used? (e.g., semantic versioning) 

8. Model Deployment:    
- What is the delivery format for the model?  
- What is the expected time for changes? (Time from commit to production) 
- What is the target environment to serve predictions?    
- What is your model release policy? Is A/B testing or multi-armed bandits testing required? (e.g., for measuring the effectiveness of the new model on business metrics and deciding what model should be promoted into the production environment)  
- What is your deployment strategy? (e.g. shadow/canary deployment required?) 

9. Prediction Serving:  
- What is the serving mode? (batch or online) 
- Is distributed model serving required?  
- Is multi-model prediction serving required? 
- Is pre-assertion for input data implemented?    
- What fallback method for an inadequate model output (post-assertion) is implemented? (Do we have a heuristic benchmark?)    
- Do you need ML inference accelerators (TPUs)?   
- What is the expected target volume of predictions per month or hours?  

10. ML Model, Data, and System Monitoring:   
- Is this optional or mandatory? For instance, do you need to assess the effectiveness of your model during prediction serving? Do you need to monitor your model for performance degradation and trigger an alert if your model starts performing badly? Is the model retraining based on events such as data or concept drift?  
- What ML metrics are collected?  
- What domain-specific metrics are collected? 
- How is the model performance decay detected? (Data Monitoring)  
- How is the data skew detected? (Data Monitoring)    
- What operational aspects need to be monitored? (e.g., model prediction latency, CPU/RAM usage)  
- What is the alerting strategy? (thresholds) 
- What triggers the model re-training? (ad hoc, event-based, or scheduled)    

11. Metadata Store: 
- What kind of metadata in code, data, and model management need to be collected? (e.g., the pipeline run ID, trigger, performed steps, start/end timestamps, train/test dataset split, hyperparameters, model object, various statistics/profiling, etc.)  
- Are any ML governance processes included in the MLOps lifecycle? What metadata will be required?  
- What is the documentation strategy: Do we treat documentation as a code? (examples: Datasheets for Datasets and Model Card for Model Reporting)   
- What operational metrics need to be collected? E.g., time to restore, change fail percentage.  

## HOW TO RUN