import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data():
    """Load the Titanic dataset."""
    return pd.read_csv('data/titanic.csv')

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Create copy of dataframe
    df = df.copy()
    
    # Handle missing values
    numeric_features = ['Age', 'Fare']
    imputer = SimpleImputer(strategy='median')
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    
    # Feature engineering
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Drop unnecessary columns
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'])
    
    return df

if __name__ == '__main__':
    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Save processed data
    processed_data.to_csv('data/processed_titanic.csv', index=False)
    print("Preprocessing completed. Data saved to processed_titanic.csv")
