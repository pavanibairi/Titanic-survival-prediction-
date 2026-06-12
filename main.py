# =======================
# TITANIC SURVIVAL PREDICTION - SERVER READY VERSION
# =======================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs('plots', exist_ok=True)

# For better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*50)
print("STEP 1: DATA LOADING & UNDERSTANDING")
print("="*50)

# 1. Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

print(train_df.head())

print("\nMissing values in train:")
print(train_df.isnull().sum())

print("\n" + "="*50)
print("STEP 2: DATA PREPROCESSING + FEATURE ENGINEERING")
print("="*50)

test_df['Survived'] = -1
combined = pd.concat([train_df, test_df], ignore_index=True)

# 2.1 Feature Engineering
combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined['Title'] = combined['Title'].replace('Mlle', 'Miss')
combined['Title'] = combined['Title'].replace('Ms', 'Miss')
combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)

# 2.2 Handle Missing Values
combined['Age'] = combined.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])
combined['Fare'] = combined.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
combined = combined.drop('Cabin', axis=1)

# 2.3 Encode Categorical
le = LabelEncoder()
combined['Sex'] = le.fit_transform(combined['Sex'])
combined['Embarked'] = le.fit_transform(combined['Embarked'])
combined['Title'] = le.fit_transform(combined['Title'])

# 2.4 Drop unnecessary columns
drop_cols = ['Name', 'Ticket', 'PassengerId', 'Survived']
X = combined.drop(drop_cols, axis=1)
y = combined['Survived']

X_train = X[y!= -1]
y_train = y[y!= -1]
X_test = X[y == -1]

print(f"Final train features shape: {X_train.shape}")
print(f"Columns used: {list(X_train.columns)}")

print("\n" + "="*50)
print("STEP 3: EDA - GRAPHS SAVE TO FILES")
print("="*50)

# 3.1 Survival rate by Sex - SAVE instead of SHOW
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex - 0=Male, 1=Female')
plt.ylabel('Survival Rate')
plt.savefig('plots/survival_by_sex.png', dpi=300, bbox_inches='tight')
plt.close()
print("Graph saved: plots/survival_by_sex.png")
print("Observation: Females survival rate ~74%, Males ~19%. Sex is king feature!")

# 3.2 Survival by Pclass
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.savefig('plots/survival_by_pclass.png', dpi=300, bbox_inches='tight')
plt.close()
print("Graph saved: plots/survival_by_pclass.png")
print("Observation: 1st class survival ~63%, 3rd class ~24%. Money matters!")

# 3.3 Correlation heatmap
plt.figure(figsize=(10, 8))
corr = X_train.join(y_train).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation with Survived')
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Graph saved: plots/correlation_heatmap.png")

print("\n" + "="*50)
print("STEP 4: MODEL BUILDING")
print("="*50)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with {results[best_model_name]:.4f} accuracy")

print("\n" + "="*50)
print("STEP 5: MODEL EVALUATION")
print("="*50)

y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)[:, 1]

print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_pred):.4f}")

# Confusion Matrix - SAVE
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Graph saved: plots/confusion_matrix.png")

# ROC Curve - SAVE
fpr, tpr, thresholds = roc_curve(y_val, y_proba)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve AUC={roc_auc_val:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Graph saved: plots/roc_curve.png")

best_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.4f} with TPR-FPR diff: {tpr[best_idx]-fpr[best_idx]:.4f}")

print("\n" + "="*50)
print("STEP 6: PREDICTION & SUBMISSION")
print("="*50)

best_model.fit(X_train, y_train)
test_pred = best_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_pred.astype(int)
})
submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")
print(submission.head(10))

print("\n" + "="*50)
print("ALL GRAPHS SAVED IN 'plots' FOLDER!")
print("="*50)