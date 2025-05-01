import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Download Titanic dataset from a URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Data preprocessing
def preprocess_data(df):
    # Create a copy of the dataframe
    data = df.copy()
    
    # Handle missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Convert categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = pd.concat([data, embarked_dummies], axis=1)
    
    # Select features for model
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked_C', 'Embarked_Q', 'Embarked_S']
    
    return data[features], data['Survived']

# Prepare the data
X, y = preprocess_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate multiple models
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy, model

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate all models
results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    accuracy, trained_model = train_evaluate_model(model, X_train_scaled, X_test_scaled, 
                                                 y_train, y_test, name)
    results[name] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = trained_model

# Plot model comparison
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Feature importance plot for the best model
if best_model:
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else np.zeros(len(X.columns))
    }).sort_values('importance', ascending=False)

    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Predicting Survival')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Save results to statistics file
with open('statistics.txt', 'w') as f:
    f.write("Dataset Statistics:\n")
    f.write(f"Total passengers: {len(df)}\n")
    f.write(f"Survival rate: {df['Survived'].mean():.2%}\n")
    f.write(f"Average age: {df['Age'].mean():.1f} years\n")
    f.write(f"Percent female: {(df['Sex'] == 'female').mean():.2%}\n")
    f.write("\nModel Performance Comparison:\n")
    for model_name, accuracy in results.items():
        f.write(f"{model_name}: {accuracy:.4f}\n")
