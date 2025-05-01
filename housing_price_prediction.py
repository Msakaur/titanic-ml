import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

def preprocess_data(df):
    # Create a copy of the dataframe
    data = df.copy()
    
    # Handle any missing values if present
    for column in data.columns:
        if data[column].isnull().any():
            data[column].fillna(data[column].median(), inplace=True)
    
    return data

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    print(f"\n{model_name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Average CV R² score: {cv_scores.mean():.4f}")
    
    return rmse, r2, model

# Preprocess the data
data = preprocess_data(df)

# Split features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0)
}

# Train and evaluate all models
results = {}
best_model = None
best_r2 = -float('inf')

for name, model in models.items():
    rmse, r2, trained_model = train_evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test, name
    )
    results[name] = {'RMSE': rmse, 'R2': r2}
    if r2 > best_r2:
        best_r2 = r2
        best_model = trained_model

# Plot model comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
rmse_values = [results[name]['RMSE'] for name in model_names]
r2_values = [results[name]['R2'] for name in model_names]

# Create subplot for RMSE
plt.subplot(1, 2, 1)
plt.bar(model_names, rmse_values)
plt.title('Model Comparison - RMSE')
plt.xticks(rotation=45)
plt.ylabel('RMSE')

# Create subplot for R²
plt.subplot(1, 2, 2)
plt.bar(model_names, r2_values)
plt.title('Model Comparison - R²')
plt.xticks(rotation=45)
plt.ylabel('R² Score')

plt.tight_layout()
plt.savefig('regression_model_comparison.png')
plt.close()

# Feature importance analysis
if hasattr(best_model, 'coef_'):
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.coef_)
    }).sort_values('importance', ascending=False)

    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Predicting Housing Prices')
    plt.tight_layout()
    plt.savefig('regression_feature_importance.png')
    plt.close()

# Save results to statistics file
with open('regression_statistics.txt', 'w') as f:
    f.write("California Housing Dataset Statistics:\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Average house price: ${y.mean():.2f}\n")
    f.write("\nModel Performance Comparison:\n")
    for model_name, metrics in results.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"R² Score: {metrics['R2']:.4f}\n")
