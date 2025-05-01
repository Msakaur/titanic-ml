# Titanic Survival Prediction

This project demonstrates a machine learning approach to predict survival on the Titanic using the famous Titanic dataset. The implementation uses Random Forest Classifier to predict passenger survival based on various features.

## Features Used
- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Number of Siblings/Spouses (SibSp)
- Number of Parents/Children (Parch)
- Fare
- Port of Embarkation (Embarked)

## Project Structure
- `titanic_analysis.py`: Main script containing the analysis and model
- `requirements.txt`: List of Python dependencies
- `feature_importance.png`: Generated plot showing feature importance
- `statistics.txt`: Basic dataset statistics

## Setup and Running
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```
   python titanic_analysis.py
   ```

## Model Performance
The script will output:
- Model accuracy
- Detailed classification report
- Feature importance visualization
- Basic dataset statistics

## Dataset
The dataset is automatically downloaded from a public URL containing the Titanic passenger data.
