# Diabetes Prediction Using Artificial Neural Network (ANN)
This project predicts the likelihood of diabetes using the Pima Indian Diabetes dataset with an Artificial Neural Network (ANN) for binary classification. The model is trained, evaluated, and then used for future predictions.


## Introduction
The project uses ANN to classify diabetes presence based on medical attributes from the Pima Diabetes dataset.

## Project Workflow
1. Load and preprocess data.
2. Split the data into training and testing sets.
3. Train the ANN model.
4. Evaluate model performance using accuracy, recall, and precision.
5. Save the trained model and scaler for future use.


## Usage
### Training the Model
Run `diabetes_train.py` to preprocess the data, train the model, and save the model and scaler:
   `python diabetes_train.py`
### Making Predictions
Use `diabetes_predict.py` to load the model and make predictions:
   `python diabetes_predict.py`
