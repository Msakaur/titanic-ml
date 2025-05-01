import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_processed_data():
    """Load the processed dataset."""
    return pd.read_csv('data/processed_titanic.csv')

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models."""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        with open(f'models/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred)
        }
    
    return results

if __name__ == '__main__':
    # Load data
    data = load_processed_data()
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Save results
    with open('results/statistics.txt', 'w') as f:
        for model_name, result in results.items():
            f.write(f"\n{model_name.upper()} RESULTS:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Classification Report:\n{result['report']}\n")
