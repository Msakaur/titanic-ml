# Titanic ML Project

This project implements machine learning models to predict survival on the Titanic.

## Project Structure

```
titanic_ml/
├── data/               # Raw dataset
├── models/            # Saved trained models
├── results/           # Outputs (plots, reports)
├── scripts/           # Code files
├── README.md          # Project overview
└── requirements.txt   # Python dependencies
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place the Titanic dataset in the `data/` directory.

## Usage

1. Data preprocessing:
   ```
   python scripts/preprocess.py
   ```

2. Train models:
   ```
   python scripts/train.py
   ```

3. Generate visualizations:
   ```
   python scripts/visualize.py
   ```
