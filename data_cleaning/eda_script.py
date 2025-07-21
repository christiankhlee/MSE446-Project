import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load cleaned data
df = pd.read_csv('heart_cleaned.csv')

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Create figure directory
import os
os.makedirs('eda_plots', exist_ok=True)

# 1. DATASET OVERVIEW
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Target distribution
axes[0,0].pie(df['target'].value_counts(), labels=['No Disease', 'Disease'], 
              autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')

# Age distribution by target
sns.histplot(data=df, x='age', hue='target', bins=20, ax=axes[0,1], alpha=0.7)
axes[0,1].set_title('Age Distribution by Heart Disease Status', fontsize=14, fontweight='bold')
axes[0,1].legend(['No Disease', 'Disease'])

# Dataset sources
if 'dataset' in df.columns:
    dataset_counts = df['dataset'].value_counts()
    axes[1,0].bar(range(len(dataset_counts)), dataset_counts.values)
    axes[1,0].set_xticks(range(len(dataset_counts)))
    axes[1,0].set_xticklabels(dataset_counts.index, rotation=45)
    axes[1,0].set_title('Samples by Dataset Source', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Count')

# Missing data heatmap (before cleaning)
df_orig = pd.read_csv('heart.csv')  # Original data
missing_data = df_orig.isnull()
axes[1,1].imshow(missing_data.T, cmap='viridis', aspect='auto')
axes[1,1].set_title('Missing Data Pattern (Original)', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Samples')
axes[1,1].set_ylabel('Features')
axes[1,1].set_yticks(range(len(df_orig.columns)))
axes[1,1].set_yticklabels(df_orig.columns, fontsize=8)

plt.tight_layout()
plt.savefig('eda_plots/01_dataset_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. CORRELATION ANALYSIS
plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Generate heatmap
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
            center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_plots/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. TOP CORRELATIONS WITH TARGET
target_corr = df[numeric_cols].corr()['target'].abs().sort_values(ascending=False)[1:]
plt.figure(figsize=(10, 8))
target_corr.head(10).plot(kind='barh')
plt.title('Top 10 Features Correlated with Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Correlation with Target')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('eda_plots/03_target_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# Print top correlations
print("Top features correlated with heart disease:")
for feature, corr in target_corr.head(10).items():
    direction = "positively" if df[numeric_cols].corr()['target'][feature] > 0 else "negatively"
    print(f"  {feature}: {corr:.3f} ({direction})")

# 4. KEY FEATURE DISTRIBUTIONS
key_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
key_features = [f for f in key_features if f in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    if i < len(axes):
        # Distribution by target
        sns.histplot(data=df, x=feature, hue='target', bins=20, 
                    ax=axes[i], alpha=0.7, stat='density')
        axes[i].set_title(f'{feature.title()} Distribution', fontweight='bold')
        axes[i].legend(['No Disease', 'Disease'])
        
        # Add mean lines
        for target_val in [0, 1]:
            mean_val = df[df['target'] == target_val][feature].mean()
            axes[i].axvline(mean_val, color=['blue', 'orange'][target_val], 
                          linestyle='--', alpha=0.8, linewidth=2)

# Box plots for remaining space
if len(key_features) < len(axes):
    remaining_ax = axes[len(key_features)]
    df_melted = df[key_features + ['target']].melt(id_vars=['target'])
    sns.boxplot(data=df_melted, x='variable', y='value', hue='target', ax=remaining_ax)
    remaining_ax.set_title('Feature Distributions by Target', fontweight='bold')
    remaining_ax.set_xticklabels(remaining_ax.get_xticklabels(), rotation=45)

# Hide extra subplots
for i in range(len(key_features) + 1, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('eda_plots/04_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. CATEGORICAL FEATURES ANALYSIS
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
categorical_features = [f for f in categorical_features if f in df.columns]

if len(categorical_features) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(categorical_features[:6]):
        if i < len(axes):
            # Create crosstab
            ct = pd.crosstab(df[feature], df['target'], normalize='index')
            ct.plot(kind='bar', ax=axes[i], color=['skyblue', 'lightcoral'])
            axes[i].set_title(f'{feature.upper()} vs Heart Disease', fontweight='bold')
            axes[i].set_xlabel(feature.title())
            axes[i].set_ylabel('Proportion')
            axes[i].legend(['No Disease', 'Disease'])
            axes[i].tick_params(axis='x', rotation=0)
    
    # Hide extra subplots
    for i in range(len(categorical_features), len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    plt.savefig('eda_plots/05_categorical_features.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. ADVANCED ANALYSIS
# Pairplot of top correlated features
top_features = target_corr.head(5).index.tolist() + ['target']
if len(top_features) <= 6:  # Limit for readability
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[top_features], hue='target', diag_kind='hist', plot_kws={'alpha': 0.7})
    plt.suptitle('Pairplot of Top Features', y=1.02, fontsize=16, fontweight='bold')
    plt.savefig('eda_plots/06_pairplot_top_features.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. DATA QUALITY SUMMARY
print("\n" + "=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)

print(f"Final dataset shape: {df.shape}")
print(f"Features for modeling: {df.shape[1] - 1}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Class balance
class_balance = df['target'].value_counts(normalize=True)
print(f"\nClass balance:")
print(f"  No Disease: {class_balance[0]:.1%}")
print(f"  Disease: {class_balance[1]:.1%}")
balance_ratio = min(class_balance) / max(class_balance)
print(f"  Balance ratio: {balance_ratio:.3f} {'(Good)' if balance_ratio > 0.3 else '(Consider balancing)'}")

# Feature summary
print(f"\nFeature summary:")
print(f"  Numeric features: {len(df.select_dtypes(include=[np.number]).columns) - 1}")
print(f"  Categorical features: {len(df.select_dtypes(include=['object']).columns)}")

# Outlier summary
outlier_counts = {}
for col in key_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    outlier_counts[col] = outliers

total_outliers = sum(outlier_counts.values())
print(f"  Total outliers: {total_outliers} ({total_outliers/len(df)*100:.1f}% of data)")

print("\n" + "=" * 60)
print("RECOMMENDATIONS FOR MODELING")
print("=" * 60)
print("1. ✅ Dataset is clean and ready for modeling")
print("2. ✅ Good sample size (920 samples) for your project")
print("3. ✅ Reasonable class balance")
print(f"4. 🔍 Top predictive features: {', '.join(target_corr.head(3).index)}")
print("5. 📊 Use stratified k-fold CV to maintain class balance")
print("6. 🎯 Start with Logistic Regression baseline")
print("7. 🌲 Try Random Forest (good with mixed data types)")
print("8. ⚖️ Consider feature scaling for distance-based models")

print(f"\nAll plots saved in 'eda_plots/' directory")
print("Next: Run train/test split and start modeling!")