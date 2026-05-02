import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def get_unnecessary_cols(df):
    return [col for col in df.columns if col.lower() in [
        'id', 'timestamp', 'flow id', 'src ip', 'dst ip', 'src port', 'dst port', 'protocol'
    ]]

def preprocess_data(df, is_training=False, prep_artifacts=None):
    df = df.copy()
    
    # 1. Drop unnecessary columns
    unnecessary_cols = get_unnecessary_cols(df)
    df.drop(columns=unnecessary_cols, inplace=True, errors='ignore')
    
    has_label = False
    y = None
    
    if is_training:
        if 'Label' not in df.columns:
            target_col = df.columns[-1]
            df.rename(columns={target_col: 'Label'}, inplace=True)
        y = df['Label']
        X = df.drop('Label', axis=1)
        expected_features = X.columns.tolist()
        
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        num_imputer = SimpleImputer(strategy='median')
        if numeric_cols:
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
            
        cat_imputer = SimpleImputer(strategy='most_frequent')
        label_encoders = {}
        if categorical_cols:
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                
        y_encoder = LabelEncoder()
        y_encoded = y_encoder.fit_transform(y.astype(str))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        prep_artifacts = {
            'expected_features': expected_features,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'label_encoders': label_encoders,
            'y_encoder': y_encoder,
            'scaler': scaler
        }
        
        return X_scaled, y_encoded, prep_artifacts
        
    else:
        # Inference mode
        if prep_artifacts is None:
            raise ValueError("prep_artifacts must be provided for inference")
            
        if 'Label' in df.columns:
            has_label = True
            y = df['Label']
            X = df.drop('Label', axis=1)
        else:
            expected_features = prep_artifacts['expected_features']
            if len(df.columns) > len(expected_features) and df.columns[-1] not in expected_features:
                 has_label = True
                 y = df.iloc[:, -1]
                 X = df.iloc[:, :-1]
            else:
                 X = df
                 
        expected_features = prep_artifacts['expected_features']
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        
        numeric_cols = prep_artifacts['numeric_cols']
        categorical_cols = prep_artifacts['categorical_cols']
        
        if numeric_cols:
            X[numeric_cols] = prep_artifacts['num_imputer'].transform(X[numeric_cols])
            
        if categorical_cols:
            X[categorical_cols] = prep_artifacts['cat_imputer'].transform(X[categorical_cols])
            for col in categorical_cols:
                le = prep_artifacts['label_encoders'][col]
                mapping = {v: k for k, v in enumerate(le.classes_)}
                X[col] = X[col].astype(str).map(mapping).fillna(0).astype(int)
                
        X_scaled = prep_artifacts['scaler'].transform(X)
        
        if has_label:
            y_encoder = prep_artifacts['y_encoder']
            mapping = {v: k for k, v in enumerate(y_encoder.classes_)}
            y_encoded = y.astype(str).map(mapping).fillna(0).astype(int)
            return X_scaled, y_encoded
        else:
            return X_scaled, None
