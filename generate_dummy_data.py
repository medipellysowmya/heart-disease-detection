import pandas as pd
import numpy as np
import os

# Define schema
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

# Generate dummy data
n_rows = 100
data = {
    'age': np.random.randint(20, 80, n_rows),
    'sex': np.random.randint(0, 2, n_rows),
    'cp': np.random.randint(0, 4, n_rows),
    'trestbps': np.random.randint(90, 200, n_rows),
    'chol': np.random.randint(100, 600, n_rows),
    'fbs': np.random.randint(0, 2, n_rows),
    'restecg': np.random.randint(0, 3, n_rows),
    'thalach': np.random.randint(60, 220, n_rows),
    'exang': np.random.randint(0, 2, n_rows),
    'oldpeak': np.random.rand(n_rows) * 6,
    'slope': np.random.randint(0, 3, n_rows),
    'ca': np.random.randint(0, 4, n_rows),
    'thal': np.random.randint(0, 4, n_rows),
    target: np.random.randint(0, 2, n_rows)
}

df = pd.DataFrame(data)

# Ensure directory exists
os.makedirs('Notebook_Experiments/Data', exist_ok=True)

# Save to CSV
df.to_csv('Notebook_Experiments/Data/heart.csv', index=False)
print("Dummy data created at Notebook_Experiments/Data/heart.csv")
