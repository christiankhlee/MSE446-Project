# Data Cleaning and Exploratory Data Analysis (EDA)

This folder contains scripts and outputs for the data cleaning and exploratory data analysis (EDA) phase of the heart disease prediction project.

## Contents

- `cleaning_script.py` – Cleans the raw dataset (`heart.csv`) by imputing missing values, encoding categorical features, creating a binary target variable, and saving the cleaned dataset as `heart_cleaned.csv`.
- `eda_script.py` – Generates summary statistics and EDA visualizations based on the cleaned dataset.
- `heart_cleaned.csv` – Output of the cleaning script.
- `eda_plots/` – Directory containing all EDA plots as `.png` files.

## Running the Scripts

1. Place the raw dataset file (`heart.csv`) in this folder.
2. Run the cleaning script:
python cleaning_script.py

markdown
Copy
Edit
This produces `heart_cleaned.csv`.

3. Run the EDA script:
python eda_script.py

markdown
Copy
Edit
This saves all EDA plots in the `eda_plots/` folder and prints a data quality summary to the console.

## Key Outputs and Analysis

### Cleaned Dataset
- **File:** `heart_cleaned.csv`
- 920 rows, 16 columns, no missing values. Target variable converted to binary (`0`: no disease, `1`: disease). Categorical features encoded as integers. Outliers retained.

### EDA Plots

#### 1. Dataset Overview
**File:** `01_dataset_overview.png`  
- **Heart Disease Distribution:** About 55% of patients have heart disease, while 45% do not, indicating reasonable class balance.  
- **Age Distribution by Status:** Patients with heart disease tend to skew slightly older.  
- **Samples by Dataset Source:** Data comes from four different sources, with most samples from datasets `0` and `1`.  
- **Missing Data Pattern:** Highlights significant missingness in columns like `ca` and `thal` in the original data before cleaning.

#### 2. Feature Correlation Matrix
**File:** `02_correlation_matrix.png`  
- Shows pairwise correlations between all numeric features.  
- `target` is moderately correlated (both positively and negatively) with several features, suggesting predictive potential.  
- Some features, like `ca` and `thal`, show strong internal correlations.

#### 3. Top Features Correlated with Heart Disease
**File:** `03_target_correlations.png`  
- Identifies the ten features most correlated with heart disease.  
- `cp`, `thalch`, and `exang` are the strongest predictors, with `cp` and `thalch` negatively correlated and `exang` positively correlated with disease.

#### 4. Feature Distributions
**File:** `04_feature_distributions.png`  
- Shows distributions of key numeric features (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`) by target.  
- For example, higher `oldpeak` and lower `thalch` values are associated with heart disease.  
- Differences in means between groups are visible and suggest these features are informative.

#### 5. Categorical Feature Distributions
**File:** `05_categorical_features.png`  
- Compares proportions of heart disease across categories of features like `sex`, `cp`, `fbs`, `restecg`, `exang`, and `slope`.  
- Notably, male patients (`sex=1`) have a higher incidence of heart disease, and certain chest pain types (`cp`) are much more common in patients with disease.

#### 6. Pairplot of Top Features
**File:** `06_pairplot_top_features.png`  
- Visualizes pairwise relationships among the top correlated features and their distributions.  
- Shows how patients with and without heart disease cluster differently in feature space, reinforcing which features are most predictive.

## Notes

- All cleaning decisions preserved the dataset size while making it usable for modeling.
- Outliers have been flagged but not removed at this stage.
- EDA highlights that several features show clear separations between disease and no disease groups, which will guide feature selection and modeling.