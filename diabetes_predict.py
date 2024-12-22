import pandas as pd
from tensorflow.keras.models import load_model
import pickle  # load scaler

# Load model
model = load_model('diabetes-model.h5')

# Load scaler
with open('diabetes-scaler_x.pickle', 'rb') as f:
    scaler_x = pickle.load(f)

# Load the same dataset for prediction
new_data = pd.read_csv('diabetes.csv')

# Exclude the target variable 'Outcome' from features
features_for_prediction = new_data.drop('Outcome', axis=1)

# Pre-process the new data
features_scaled = scaler_x.transform(features_for_prediction)

# Make predictions
predictions = model.predict(features_scaled)
predictions_class = predictions > 0.5

# Print predictions
for i in range(len(predictions)):
    print(f'{features_for_prediction.iloc[i]}\nOutcome: {predictions_class[i][0]}\n')
