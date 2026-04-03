import pandas as pd
import numpy as np

# Load the data
print("Looking for CSV files...")
import glob
csv_files = glob.glob("data/*.csv") + glob.glob("*.csv")
print(f"Found CSV files: {csv_files}")

if csv_files:
    file_path = csv_files[0]
    print(f"\nLoading: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nColumn names:")
    print(df.columns.tolist())
    
    print(f"\nCLASS_LABEL value counts:")
    if 'CLASS_LABEL' in df.columns:
        print(df['CLASS_LABEL'].value_counts())
        print(f"\nMissing values in CLASS_LABEL: {df['CLASS_LABEL'].isnull().sum()}")
    else:
        print("CLASS_LABEL column not found!")
        print("Looking for similar columns...")
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                print(f"  Found: {col}")
    
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
else:
    print("No CSV files found!")
