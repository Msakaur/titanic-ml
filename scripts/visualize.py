import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def plot_feature_importance():
    """Plot feature importance from Random Forest model."""
    # Load model and data
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    data = pd.read_csv('data/processed_titanic.csv')
    
    # Get feature importance
    features = data.drop('Survived', axis=1).columns
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')

def plot_model_comparison():
    """Plot model comparison."""
    # Read statistics
    with open('results/statistics.txt', 'r') as f:
        content = f.read()
    
    # Extract accuracies
    accuracies = {}
    for line in content.split('\n'):
        if 'Accuracy:' in line:
            model = line.split('RESULTS')[0].strip().lower()
            accuracy = float(line.split(':')[1])
            accuracies[model] = accuracy
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')

if __name__ == '__main__':
    plot_feature_importance()
    plot_model_comparison()
    print("Visualizations generated successfully.")
