# Hearth Failure Prediction using Azure Machine Learning

This project is a capstone project for Udacity's Machine Learning Engineer for Microsoft Azure Nanodegree. In this project, a classification model was trained to predict the event of hearth failure. There were two method used to create the model which were Azure Automated ML and Hyperdrive Run. 

The performance of both models were compared and the best model was choosen based on the accuracy. This model then deployed using Azure Container Instances (ACI) and can be consumed using HTTP request through REST endpoint.

The overall process of this project can be seen in this diagram:
![Workflow]()  
*Figure 1: Project Workflow


## Project Set Up and Installation
This project was done in Azure Environment provided by Udacity. Overall, this project use:
- Jupyter lab
- Python 3.6
- Azure ML Studio
- Azure ML SDK 

## Dataset

### Overview
This project use Hearth Failure Dataset from Kaggle that can be acquired from this [link](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). This dataset contains thirteen columns related to hearth failure.

### Task
The occurence of death was predicted using twelve features outlined in the dataset. These features were:
- age (int): self explanatory
- anaemia (bool): whether there has been a decrease of red blood cells or hemoglobin
- creatinine_phosphokinase (int): level of the CPK enzyme in the blood in mcg/L
- diabetes (bool): whether the patient has diabetes
- ejection_fraction (int): percentage of blood leaving the heart at each contraction
- high_blood_pressure (bool): whether the patient has hypertension
- platelets (int): platelets in the blood in kiloplatelets/mL
- serum_creatinine (float): level of serum creatinine in the blood in mg/dL
- serum_sodium (int): level of serum sodium in the blood in mEq/L
- sex (int): female or male (binary)

### Access
This dataset can be accessed inside dataset folder or from this [link]().

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
