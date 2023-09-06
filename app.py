import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from keras.layers import Embedding

# Load your dataset (replace 'your_data.csv' with your actual dataset)
data = pd.read_csv('sample_submission.csv')

# Split data into features (X) and target (y)
X = data.drop(columns=['Age'])
y = data['Age']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# processing train and test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build an ANN model
model = keras.Sequential([# Input Layer
                          layers.Dense(units=10, activation='relu', input_dim=X_train.shape[1]),

                          # Hidden Layers
                          #hidden layer 1
                          layers.Dense(200,activation='relu'),
                          # Hidden layer 2
                          layers.Dense(200,activation='relu'),
                          # Hidden layer 3
                          layers.Dense(200,activation='relu'),

                          # Output Layer
                          layers.Dense(1),
                         ])
early_stoping = keras.callbacks.EarlyStopping(patience=20,
                                              min_delta=0.001,
                                              restore_best_weights=True,
                                             )
# Compile the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
              )
# Train the model
model.fit(X_train,y_train,
                      validation_data=(X_test,y_test),  # validation data
                      batch_size = 516,                 # number of inputs for optimizer
                      epochs = 200,                     # number of iteration
                      callbacks=[early_stoping],        # callback to stop iteration when val loss stops decreasing
                      verbose = 0
                      )

# Streamlit UI
st.title('Crab Age Detection App using ANN')
st.write("Enter the features below to make a prediction:")

# Input form for user
input_features = {}
for feature in X.columns:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_features[feature] = value

if st.button('Predict'):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([input_features])
    input_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data)[0][0]

    st.write(f"Predicted Crab Age: {prediction:.2f} years")

# Optionally, you can add evaluation metrics for your model
st.write("Model Evaluation:")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")
