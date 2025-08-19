# Imports
import pandas as pd # Library for data loading, cleaning, and manipulation (DataFrames, CSVs)
import numpy as np # Library for numerical operations (arrays, maths)
import matplotlib.pyplot as plt # Core python plotting library
import seaborn as sns # Statistical plotting built on Matplotlib (~ nicer visuals)
import joblib # For saving and loading trained models efficiently
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # train_test_split = splits dataset into training and testing sets. cross_val_score = runs cross-validation to test model performance. GridSearchCV = tunes hyperparameters by testing multiple combinations.
from sklearn.preprocessing import StandardScaler # Standardizes features (mean = 0, std = 1)
from sklearn.linear_model import LogisticRegression # The actial ml model being used
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc) # metrics to value accuracy of model
from sklearn.pipeline import Pipeline # chains preprocessing + model into one object

# This is where the csv file is located ~ the path to the file:
DATA_PATH = r'<Path to dataset>'

# Load dataset
df = pd.read_csv(DATA_PATH) # reads the csv file into a panda dataframe
print('Shape:', df.shape) # .shape is a tuple (#row, #columns)
print(df.head()) # .head() prints the first 5 rows
print('\nColumns:', df.columns.tolist()) # .columns.tolist() lists column names
print('\nMissing values:\n', df.isnull().sum()) # .isnull().sum() shows missing values per column


from sklearn.impute import SimpleImputer

# Drop unwanted columns first
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32']) # Drops "Unnamed: 32" if present -- common leftover from csv export with extra commas

# Map diagnosis to target if needed
if 'diagnosis' in df.columns and 'target' not in df.columns:
    df['target'] = df['diagnosis'].map({'B':0, 'M':1}) # Maps B (Benign) -> 0 and M (Malignant) -> 1

# Drop ID columns if present
for col in ['id', 'Id', 'ID']:
    if col in df.columns:
        df.drop(columns=col, inplace=True) # Removes idenitifiers that don't help prediction

# Define features and target
X = df.drop(columns=['diagnosis', 'target'], errors='ignore') # X = features for model training
y = df['target'] # y = target labels (0 or 1)

# Impute missing values in X (handles missing values)
#(i.e. fills missing values with column means)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Now split into train/test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
) # Splits data -> 80% train and 20% test. ("stratify=y" keeps the class proportions the same.)

# Scale features (ensures all features are on the same scale)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now you can fit your model without errors
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=10000, penalty='l2', solver='liblinear', random_state=42)
clf.fit(X_train_scaled, y_train)
        # creates Logistic Regression model
        # "max_iter=1000" avoids convergence warnings
        # "penalty='12'" uses Ridge regularization
        # "solver='liblinear'" is efficient for small datasets
        
# Basic EDA (Exploratory Data Analysis)
print('\nValue counts for diagnosis (raw):')
if 'diagnosis' in df.columns: # checks whether the dataframe "df" has a column named "diagnosis". This decides whether to print the diagnosis counts.
    print(df['diagnosis'].value_counts()) # if diagnosis exists, print counts of each class (e.g. how many 'B' and 'M'). Quick sanity check for class balance.
else: # start of the alternative branch if diagnosis column is missing
    print('No diagnosis column found. Columns:', df.columns) # If no diagnosis column, print the message and list all column names so you can debug whats in the dataset.

# If 'diagnosis' exists map to target
if 'diagnosis' in df.columns and 'target' not in df.columns: # checks both that diagnosis exists and target does not yet exist - prevents remapping twice.
    df['target'] = df['diagnosis'].map({'B':0, 'M':1}) # Creates a new column target with numeric labels: benign 'B' -> 0, malignant 'M' -> 1

# Drop ID columns if present
for col in ['id', 'Id', 'ID']: # loop over common ways to an ID column might be named (case varients)
    if col in df.columns: # check whether the current name exists as a column
        df.drop(columns=col, inplace=True) # if present, permanently drop that ID column. IDs don't carry predictive info, can leak or hurt model training.

# Quick stats and correlation
print("Plot 1: Diagnosis class distribution") # prints a short label so console output is readable and you know which plot is generated.
print(df.describe()) # print summary statistics (count, mean, std, percentiles) for numeric columns - quick data overview.
plt.figure(figsize=(10,6)) # create a new matplotlib figure sized 10x6 inches for the count plot.
sns.countplot(x='diagnosis', data=df) # uses Seaborn to draw a bar chart of how many samples per diagnosis class (B vs M)
plt.title('Class distribution (diagnosis)') # Sets the plot title


# Correlation heatmap of features (numeric only)
print("Plot 2: Correlation heatmap") # prints the plot title in the terminal
numeric = df.select_dtypes(include=[np.number]).columns.tolist() # find all numeric columns in the df and stores their names in "numeric". Heatmap should only use numbers.
plt.figure(figsize=(14,12)) # creates a larger figure (14x12 inches) for the correlation matrix
sns.heatmap(df[numeric].corr(), cmap='coolwarm', linewidths=0.3) # compute correlations among numeric columns (df[numeric].corr()) and plot them as heatmap. "cmap" controls colours; "linewidths" add thin lines between cells.
plt.title('Correlation matrix (numeric features)') # title for the heatmap


# Preprocessing: prepare X and y
if 'target' not in df.columns: # check that the "target" column exists; if not, script raises an error because later code expects it.
    raise KeyError("No 'target' column. Ensure you have a 'diagnosis' column with values B and M.") # throw a clear error describing what's missing so you can fix the dataset.

X = df.drop(columns=['diagnosis','target'], errors='ignore') # create X by dropping diagnosis and target columns - X should contain only features. "errors='ignore'" means "don't crash if column not present."
# If other non-feature columns exist, you can drop them here
# e.g., drop columns with low variance or irrelevant metadata if present

y = df['target'] # set y to the target vector (0/1 labels). This is what the model will learn to predict.

# Check class balance
print('\nTarget distribution:') # prints a header for the upcoming ditribution output.
print(y.value_counts(normalize=True)) # shows proportion of each class. 

# Train/test split (stratify to keep class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)   # train_test_split really does do what its name is; splits features and labels into train and test sets.
    # Paramters: use 20% test set, 'random_state=42' ensures reproducible split, 'stratify=y' preserves class proportions.
print('\nTrain shape:', X_train.shape, 'Test shape:', X_test.shape) # print shapes of train and test data so that you can confirm expected sizes and number of features.

# Standardize features using a pipeline later; here we prepare scaler fit
scaler = StandardScaler() # Instantiate a StandardScaler object which will standardize features to mean 0 and std 1.
X_train_scaled = scaler.fit_transform(X_train) #
X_test_scaled = scaler.transform(X_test)

# Baseline Logistic Regression
clf = LogisticRegression(max_iter=10000, penalty='l2', solver='liblinear', random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:,1]

print('\nClassification report (Test set):')
print(classification_report(y_test, y_pred, digits=4))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}')

# Confusion matrix
print("Plot 3: Confusion matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


# ROC curve
print("Plot 4: ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_val:.4f}')
plt.plot([0,1],[0,1],'k--', alpha=0.6)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Baseline Logistic Regression)')
plt.legend()


# Coefficients
print("Plot 5: Coefficients")
coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(8,10))
coefs.plot(kind='barh')
plt.title('Logistic Regression Coefficients (baseline)')



# Hyperparameter tuning with pipeline and GridSearchCV
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=10000, random_state=42))])

param_grid = {
    'clf__penalty': ['l1','l2'],
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__solver': ['liblinear','saga']
}

# Note: 'saga' supports l1 and l2 for LogisticRegression, but requires proper scikit-learn version.
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
gs.fit(X, y)  # GridSearchCV will perform internal CV and fit pipeline (scaler included)

print('\nBest params:', gs.best_params_)
print('Best CV ROC-AUC:', gs.best_score_)

# Use best estimator to predict on test set (we re-split to ensure no data leakage)
best_model = gs.best_estimator_

# Recreate train/test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

print('\nClassification report (Best model):')
print(classification_report(y_test, y_pred, digits=4))
print('ROC-AUC (test):', roc_auc_score(y_test, y_proba))


# Interpretation: feature importance (coefficients from the logistic part of the pipeline)
# Extract classifier coefficients (after scaling, coefficients correspond to standardized features)
clf_final = best_model.named_steps['clf']
scaler_final = best_model.named_steps['scaler']

# Coefficients (since scaler is standard scaling, coefficients are comparable)
print("Plot 6: Coefficients (since scaler is standard scaling, coefficients are comparable)")
coefs = pd.Series(clf_final.coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(8,10))
coefs.plot(kind='barh')
plt.title('Feature coefficients (best model) — higher means higher log-odds of malignancy')

plt.show() # -----------> Displays all graphs simultaneously instead of one GUI a time.

# Show top positive and negative coefficients
print('\nTop features increasing odds of malignancy:')
print(coefs.sort_values(ascending=False).head(10))
print('\nTop features decreasing odds of malignancy:')
print(coefs.sort_values(ascending=True).head(10))


# Save the final pipeline (best_model) to disk
import joblib
joblib.dump(best_model, 'breast_cancer_logreg_pipeline.joblib')
print('Saved pipeline to breast_cancer_logreg_pipeline.joblib')


from sklearn.metrics import accuracy_score

# y_test = true labels, y_pred = predicted labels
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
