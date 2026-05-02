import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import traceback
from utils import preprocess_data

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        print("Error: dataset.csv not found.")
        return
        
    print("Preprocessing data...")
    X_scaled, y_encoded, prep_artifacts = preprocess_data(df, is_training=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    metrics = {}
    
    # 1. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    metrics['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist(),
        'feature_importances': rf_model.feature_importances_.tolist()
    }
    
    # 2. SVM
    print("Training SVM (with probability for confidence scores)...")
    # Using probability=True to get confidence scores (might take time on very large datasets)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    svm_pred = svm_model.predict(X_test)
    metrics['SVM'] = {
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, svm_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, svm_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, svm_pred).tolist()
    }
    
    print("\n--- Evaluation Metrics ---")
    for model_name, m in metrics.items():
        print(f"\n{model_name}:")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1-score:  {m['f1']:.4f}")
        
    print("\nSaving artifacts...")
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    with open("preprocessing.pkl", "wb") as f:
        pickle.dump(prep_artifacts, f)
    with open("metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
        
    print("End-to-end training pipeline complete. Models saved.")

if __name__ == "__main__":
    main()