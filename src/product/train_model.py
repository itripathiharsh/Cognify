import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupKFold, train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(dataset_path='data_log/model_ready_dataset.csv'):
    """
    Loads the engineered feature set, trains baseline models, evaluates them
    using per-user cross-validation, and saves the best model.
    """

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}.")
        print("Please run src/product/feature_engineering.py first to generate it.")
        return

    print(f"Dataset loaded successfully with {len(df)} sessions.")

 
    TARGET = 'self_prod_1_5'
    FEATURES = [
        col for col in df.columns if col.endswith(('_mean', '_std', '_volatility'))
        or col.startswith('emotion_hist_')
    ]
    
    if not FEATURES:
        print("Error: No feature columns found in the dataset. Check the feature engineering script.")
        return
        
    X = df[FEATURES]
    y = df[TARGET]
    groups = df['user_id']
    
    print(f"Training with {len(FEATURES)} features to predict '{TARGET}'.")

    n_splits = min(5, len(groups.unique()))
    if n_splits < 2:
        print("Warning: Not enough unique users for cross-validation. Training on all data.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"\n--- Random Forest (single user split) ---\nMAE: {mae:.4f}")

    else:
        cv = GroupKFold(n_splits=n_splits)
        print(f"Using {n_splits}-fold GroupKFold cross-validation.")
        
    
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        all_preds = []
        all_true = []
        
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            
            all_preds.extend(preds)
            all_true.extend(y_val)
            
        mae = mean_absolute_error(all_true, all_preds)
        print(f"\n--- Random Forest (Cross-Validation) ---\nMean MAE: {mae:.4f}")

    print("\nTraining final model on all available data...")
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "productivity_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    feature_importances = pd.DataFrame({
        'feature': FEATURES,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plot_path = os.path.join("assets", "feature_importance.png")
    os.makedirs("assets", exist_ok=True)
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")

if __name__ == '__main__':
    train_and_evaluate()

