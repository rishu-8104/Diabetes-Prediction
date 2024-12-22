import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
import pickle  # save encoder

# Load the diabetes dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Divide X and y
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Train and test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Build and train ANN
model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(X_test, y_test))

# Visualize training
print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict with test data
y_pred = model.predict(X_test)
y_pred_class = y_pred > 0.5

# Confusion Matrix and metrics
cm = confusion_matrix(y_test, y_pred_class)
acc = accuracy_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)

print("Confusion Matrix:")
print(cm)
print(f'Accuracy Score: {acc}')
print(f'Recall Score: {recall}')
print(f'Precision Score: {precision}')

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save model to disk
model.save('diabetes-model.h5')

# Save scaler to disk
with open('diabetes-scaler_x.pickle', 'wb') as f:
    pickle.dump(scaler_x, f)
