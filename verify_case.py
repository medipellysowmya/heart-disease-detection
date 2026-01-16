import sys
import os
import pickle
import pandas as pd
import numpy as np

def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)

# User input
input_data = {
    'age': [30],
    'sex': [1],
    'cp': [1],
    'trestbps': [140],
    'chol': [290],
    'fbs': [1],
    'restecg': [0],
    'thalach': [160],
    'exang': [0],
    'oldpeak': [1.5],
    'slope': [1],
    'ca': [0],
    'thal': [2]
}

df = pd.DataFrame(input_data)

# Load artifacts
preprocessor_path = os.path.join('Artifacts', 'Preprocessor.pkl')
model_path = os.path.join('Artifacts', 'Model.pkl')

try:
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = load_object(preprocessor_path)
    
    print(f"Loading model from {model_path}...")
    model = load_object(model_path)

    print("Transforming data...")
    transformed_data = preprocessor.transform(df)

    print("Predicting...")
    prediction = model.predict(transformed_data)

    with open("verification_result.txt", "w") as f:
        if prediction[0] == 1:
            f.write("Result: Heart Disease Detected")
        else:
            f.write("Result: No Heart Disease")
    print("Result written to verification_result.txt")

except Exception as e:
    with open("verification_result.txt", "w") as f:
        f.write(f"Error: {e}")
    print(f"Error: {e}")
