import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your data
df = pd.read_csv('heart.csv')  # Adjust filename as needed

print("=" * 60)
print("INITIAL DATA ANALYSIS")
print("=" * 60)

print(f"Dataset Shape: {df.shape}")
print(f"Total Missing Values: {df.isnull().sum().sum()}")
print(f"Missing Value Percentage: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")

# Detailed missing value analysis
print("\nMISSING VALUES ANALYSIS:")
missing_summary = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_summary.index,
    'Missing_Count': missing_summary.values,
    'Missing_Percent': missing_percent.values
}).sort_values('Missing_Percent', ascending=False)

print(missing_df[missing_df['Missing_Count'] > 0])

print("\n" + "=" * 60)
print("DATA CLEANING STRATEGY")
print("=" * 60)

# Strategy based on your missing data:
# - ca: 611/920 = 66% missing - MAJOR ISSUE
# - thal: 486/920 = 53% missing - MAJOR ISSUE  
# - slope: 309/920 = 34% missing - SIGNIFICANT
# - oldpeak, thalch, exang, trestbps, chol, fbs: <10% missing - MANAGEABLE

# Step 1: Handle columns with extreme missing values
print("Step 1: Handling columns with >50% missing values")
print("- ca: 66.4% missing")
print("- thal: 52.8% missing")
print("- slope: 33.6% missing")

# Option A: Drop these columns entirely
# Option B: Create binary indicators for missing + impute
# Option C: Use only complete cases for these features

# Let's first see what datasets you have
print(f"\nDatasets included: {df['dataset'].value_counts()}")
print(f"\nTarget variable distribution: {df['num'].value_counts()}")

# Step 2: Create a cleaned version
print("\n" + "=" * 60)
print("CLEANING PROCESS")
print("=" * 60)

# Create a copy for cleaning
df_clean = df.copy()

# Step 2a: Handle target variable (convert to binary)
print("Converting target to binary (0=no disease, 1=disease)")
df_clean['target'] = (df_clean['num'] > 0).astype(int)
print(f"New target distribution: {df_clean['target'].value_counts()}")

# Step 2b: Drop ID column (not useful for modeling)
df_clean = df_clean.drop(['id', 'num'], axis=1)

# Step 2c: Handle categorical variables
print("\nEncoding categorical variables...")
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Convert categorical columns to proper format
for col in categorical_cols:
    if col in df_clean.columns:
        print(f"  {col}: {df_clean[col].dtype} -> encoded")
        if df_clean[col].dtype == 'object':
            le = LabelEncoder()
            # Handle NaN values
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = le.fit_transform(df_clean[col])

# Step 2d: Missing value imputation strategy
print("\nImputing missing values...")

# For columns with <20% missing: impute
moderate_missing = ['trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak']
for col in moderate_missing:
    if col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Numeric: use median
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                # Categorical: use mode
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            print(f"  {col}: {missing_count} values imputed")

# For columns with high missing values: special handling
high_missing = ['slope', 'ca', 'thal']
for col in high_missing:
    if col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            # Create missing indicator
            df_clean[f'{col}_missing'] = df_clean[col].isnull().astype(int)
            # Impute with a special value or mode
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(-1, inplace=True)  # Special indicator value
            else:
                df_clean[col].fillna('missing', inplace=True)
                # Re-encode after adding 'missing'
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            print(f"  {col}: {missing_count} values handled with indicator + imputation")

print(f"\nAfter cleaning - Missing values: {df_clean.isnull().sum().sum()}")

# Step 3: Feature analysis
print("\n" + "=" * 60)
print("FEATURE ANALYSIS")
print("=" * 60)

numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('target')  # Remove target from features

print(f"Numeric features: {numeric_features}")
print(f"Total features: {len(df_clean.columns) - 1}")  # -1 for target
print(f"Final dataset shape: {df_clean.shape}")

# Step 4: Check for outliers
print("\nOutlier Analysis:")
for col in ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']:
    if col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        print(f"  {col}: {outliers} outliers ({outliers/len(df_clean)*100:.1f}%)")

# Step 5: Save cleaned dataset
df_clean.to_csv('heart_cleaned.csv', index=False)
print(f"\nCleaned dataset saved as 'heart_cleaned.csv'")

# Step 6: Basic EDA
print("\n" + "=" * 60)
print("BASIC EDA")
print("=" * 60)

# Target distribution
target_dist = df_clean['target'].value_counts()
print(f"Target distribution:")
print(f"  No Disease (0): {target_dist[0]} ({target_dist[0]/len(df_clean)*100:.1f}%)")
print(f"  Disease (1): {target_dist[1]} ({target_dist[1]/len(df_clean)*100:.1f}%)")

# Dataset source distribution
if 'dataset' in df_clean.columns:
    print(f"\nDataset sources: {df_clean['dataset'].value_counts()}")

# Correlation with target
numeric_corr = df_clean[numeric_features + ['target']].corr()['target'].sort_values(ascending=False)
print(f"\nTop features correlated with heart disease:")
print(numeric_corr[:-1].head(10))  # Exclude target itself

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("1. Review the cleaned dataset (heart_cleaned.csv)")
print("2. Run EDA visualizations")
print("3. Consider feature selection based on correlations")
print("4. Split data for training/testing") 
print("5. Scale features before modeling")
print("\nRecommended modeling approach:")
print("- Start with Logistic Regression (baseline)")
print("- Try Random Forest (handles mixed features well)")
print("- Consider ensemble methods")
print(f"- Use stratified k-fold CV (you have {len(df_clean)} samples)")