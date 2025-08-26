#!/usr/bin/env python3
"""
Generate synthetic credit card transaction data for testing the fraud detection app.
This creates a sample dataset with similar structure to the original Kaggle dataset.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_synthetic_data(n_samples=10000, fraud_rate=0.002, output_file='creditcard.csv'):
    """
    Generate synthetic credit card transaction data.
    
    Parameters:
    - n_samples: Total number of transactions
    - fraud_rate: Proportion of fraudulent transactions
    - output_file: Name of the output CSV file
    """
    print(f"🔄 Generating {n_samples} synthetic transactions...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    print(f"📊 Normal transactions: {n_normal}")
    print(f"⚠️  Fraudulent transactions: {n_fraud}")
    
    # Generate normal transactions
    normal_data = {
        'Time': np.random.uniform(0, 172800, n_normal),  # 48 hours in seconds
        'Amount': np.random.lognormal(3, 1.5, n_normal),  # Log-normal distribution
        'Class': np.zeros(n_normal, dtype=int)
    }
    
    # Add V1-V28 features (simulating PCA components)
    for i in range(1, 29):
        normal_data[f'V{i}'] = np.random.normal(0, 1, n_normal)
    
    # Generate fraud transactions (different patterns)
    fraud_data = {
        'Time': np.random.uniform(0, 172800, n_fraud),
        'Amount': np.random.lognormal(2, 2, n_fraud),  # Different amount pattern
        'Class': np.ones(n_fraud, dtype=int)
    }
    
    # Add V1-V28 features for fraud (different distribution to simulate anomalies)
    for i in range(1, 29):
        # Fraud transactions have slightly different patterns
        fraud_data[f'V{i}'] = np.random.normal(
            loc=np.random.uniform(-0.5, 0.5),  # Random offset
            scale=np.random.uniform(0.8, 1.5),  # Different variance
            size=n_fraud
        )
    
    # Combine all data
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reorder columns to match original dataset
    column_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ Dataset saved as '{output_file}'")
    print(f"📈 Dataset shape: {df.shape}")
    print(f"🔍 Fraud rate: {df['Class'].mean():.4f} ({df['Class'].sum()} fraudulent)")
    print(f"💰 Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic credit card fraud data')
    parser.add_argument('--samples', type=int, default=10000, 
                       help='Number of transactions to generate (default: 10000)')
    parser.add_argument('--fraud-rate', type=float, default=0.002,
                       help='Fraud rate (default: 0.002 = 0.2%%)')
    parser.add_argument('--output', type=str, default='creditcard.csv',
                       help='Output filename (default: creditcard.csv)')
    
    args = parser.parse_args()
    
    # Check if output file already exists
    if Path(args.output).exists():
        response = input(f"⚠️  File '{args.output}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled.")
            return
    
    # Generate the data
    df = generate_synthetic_data(
        n_samples=args.samples,
        fraud_rate=args.fraud_rate,
        output_file=args.output
    )
    
    print("\n🎉 Synthetic dataset generated successfully!")
    print("🚀 You can now run the Streamlit app:")
    print("   python3 -m streamlit run app.py")

if __name__ == "__main__":
    main()
