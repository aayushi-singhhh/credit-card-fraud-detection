# Dataset Setup Instructions

## Required Dataset: Credit Card Fraud Detection

### Option 1: Download from Kaggle
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it in the project root directory

### Option 2: Use Alternative Dataset
If you don't have access to the original dataset, you can:
1. Use any CSV with similar structure (30 features + Class column)
2. Generate synthetic data using the provided script below

### Dataset Format
The dataset should have these columns:
- `Time`: Number of seconds elapsed between each transaction and the first transaction
- `V1` to `V28`: Anonymized features (result of PCA transformation)
- `Amount`: Transaction amount
- `Class`: Target variable (0 = Normal, 1 = Fraud)

### Synthetic Data Generator
Run this Python script to create sample data for testing:

```python
import pandas as pd
import numpy as np

# Generate synthetic credit card transaction data
np.random.seed(42)

n_samples = 10000
n_fraud = int(n_samples * 0.002)  # 0.2% fraud rate

# Generate normal transactions
normal_data = {
    'Time': np.random.uniform(0, 172800, n_samples - n_fraud),  # 48 hours
    'Amount': np.random.lognormal(3, 1.5, n_samples - n_fraud),
    'Class': np.zeros(n_samples - n_fraud)
}

# Add V1-V28 features (PCA components)
for i in range(1, 29):
    normal_data[f'V{i}'] = np.random.normal(0, 1, n_samples - n_fraud)

# Generate fraud transactions
fraud_data = {
    'Time': np.random.uniform(0, 172800, n_fraud),
    'Amount': np.random.lognormal(2, 2, n_fraud),  # Different pattern
    'Class': np.ones(n_fraud)
}

# Add V1-V28 features for fraud (different distribution)
for i in range(1, 29):
    fraud_data[f'V{i}'] = np.random.normal(0.5, 1.2, n_fraud)

# Combine data
all_data = {}
for key in normal_data.keys():
    all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])

# Create DataFrame and shuffle
df = pd.DataFrame(all_data)
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv('creditcard.csv', index=False)
print(f"Generated {len(df)} transactions with {df['Class'].sum()} fraudulent cases")
```

### File Size Note
The original dataset is ~143MB and cannot be stored in GitHub. 
Make sure to download it separately and place it in your project directory.

### Quick Start
Once you have `creditcard.csv` in your project directory:
```bash
python3 -m streamlit run app.py
```
