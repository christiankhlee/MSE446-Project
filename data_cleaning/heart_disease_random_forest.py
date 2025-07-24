"""
Random Forest implementation for MSE 446 Project for Heart Disease dataset.
Group Members: Neil Patel, Lakshya Rao, Kelly Mak, Christian

Date Modified: July 24, 2025
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = 'heart_cleaned.csv'
TARGET_COLUMN = 'target'

# ---------------------------------------------------------------------------------------

#  _                    _   ____        _        
# | |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
# | |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
# | |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
# |_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|

df = pd.read_csv(DATA_FILE)

# Assign x and y data
X = df.drop(columns=[TARGET_COLUMN]) 
y = df[TARGET_COLUMN]


# Split data 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Divide the columns into numerical + categorical 
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Scale numerical columns and one-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_cols),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# ---------------------------------------------------------------------------------------

#  ____                 _                   _____                   _   
# |  _ \ __ _ _ __   __| | ___  _ __ ___   |  ___|__  _ __ ___  ___| |_ 
# | |_) / _` | '_ \ / _` |/ _ \| '_ ` _ \  | |_ / _ \| '__/ _ \/ __| __|
# |  _ < (_| | | | | (_| | (_) | | | | | | |  _| (_) | | |  __/\__ \ |_ 
# |_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_| |_|  \___/|_|  \___||___/\__|

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, 
                                          random_state=42, 
                                          class_weight='balanced'))
])

print("Training the model.")
model.fit(X_train, y_train)
print("Training complete.")

# ---------------------------------------------------------------------------------------

#  _____         _   
# |_   _|__  ___| |_ 
#   | |/ _ \/ __| __|
#   | |  __/\__ \ |_ 
#   |_|\___||___/\__|

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------------------------------------------------------------------------------

#  ____       _       _     ____                 _ _       
# |  _ \ _ __(_)_ __ | |_  |  _ \ ___  ___ _   _| | |_ ___ 
# | |_) | '__| | '_ \| __| | |_) / _ \/ __| | | | | __/ __|
# |  __/| |  | | | | | |_  |  _ <  __/\__ \ |_| | | |_\__ \
# |_|   |_|  |_|_| |_|\__| |_| \_\___||___/\__,_|_|\__|___/

print("\n" + "="*40)
print(f"Model Accuracy (on Test Set): {accuracy:.2%}")
print("="*40)

print("Classification Report:")

print(classification_report(y_test, y_pred))