#!/usr/bin/env python3
"""
Test more extreme cases to try to reproduce the user's observation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def load_model_and_data():
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv('./data/mental_health_data_final_data.csv')
    return model, df

def add_derived_features(data):
    """Add ALL the derived features from the notebook (including duplicates)"""
    data = data.copy()
    
    # First set of bracket columns
    data['Sleep_Hour_Bracket'] = pd.cut(
        data['Sleep_Hours'],
        bins=[0, 4, 5, 6, 7, 8, 9, 24],
        labels=['<4', '4–5', '5–6', '6–7', '7–8', '8–9', '9+']
    )
    data['Work_Hour_Bracket'] = pd.cut(
        data['Work_Hours'],
        bins=[0, 30, 40, 50, 60, 70, 168],
        labels=['<30', '30–40', '41–50', '51–60', '60–70', '70+']
    )
    data['Age_Bracket'] = pd.cut(
        data['Age'],
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=['<20', '21–30', '31–40', '41–50', '51–60', '60+']
    )
    data['PA_Bracket'] = pd.cut(
        data['Physical_Activity_Hours'],
        bins=[0, 1, 3, 5, 7, 10],
        labels=['0–1', '1–3', '3–5', '5–7', '7+']
    )
    data['SM_Bracket'] = pd.cut(
        data['Social_Media_Usage'],
        bins=[0, 1, 2, 3, 4, 10],
        labels=['0–1', '1–2', '2–3', '3–4', '4+']
    )
    
    # Additional bracket columns (duplicates from notebook)
    data['Sleep_Bracket'] = pd.cut(
        data['Sleep_Hours'], 
        bins=[0, 6, 8, 12], 
        labels=['<6 hrs', '6–8 hrs', '8+ hrs']
    )
    data['Work_Bracket'] = pd.cut(
        data['Work_Hours'], 
        bins=[0, 40, 55, 100], 
        labels=['<40 hrs', '40–55 hrs', '55+ hrs']
    )
    data['Activity_Bracket'] = pd.cut(
        data['Physical_Activity_Hours'], 
        bins=[0, 2, 5, 20], 
        labels=['Low', 'Moderate', 'High']
    )
    
    return data

def create_proper_preprocessor(sample_df):
    """Create the exact same preprocessing pipeline as in training"""
    feature_columns = [
        'Age', 'Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 'Severity',
        'Consultation_History', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours',
        'Social_Media_Usage', 'Diet_Quality', 'Smoking_Habit', 'Alcohol_Consumption',
        'Medication_Usage', 'Sleep_Hour_Bracket', 'Work_Hour_Bracket', 'Age_Bracket',
        'PA_Bracket', 'SM_Bracket', 'Sleep_Bracket', 'Work_Bracket', 'Activity_Bracket'
    ]
    
    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(sample_df[c])]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ], remainder="drop")
    
    preprocessor.fit(sample_df[feature_columns])
    return preprocessor, feature_columns

def test_case(name, case_data, model, preprocessor, feature_columns):
    """Test a single case"""
    case_df = pd.DataFrame([case_data])
    case_df = add_derived_features(case_df)
    
    # Transform and predict
    case_processed = preprocessor.transform(case_df[feature_columns])
    prediction = model.predict(case_processed)[0]
    probabilities = model.predict_proba(case_processed)[0]
    
    burnout_text = {0: "Low Burnout", 1: "High Burnout"}
    
    print(f"{name}:")
    print(f"  PA: {case_data['Physical_Activity_Hours']}, SM: {case_data['Social_Media_Usage']}")
    print(f"  Prediction: {burnout_text[prediction]} (confidence: {probabilities[prediction]:.1%})")
    print(f"  Probabilities: [Low: {probabilities[0]:.3f}, High: {probabilities[1]:.3f}]")
    print()
    
    return prediction, probabilities

def test_random_fallback_behavior(model):
    """Test what happens with random feature generation (as in streamlit fallback)"""
    print("=" * 60)
    print("TESTING RANDOM FALLBACK BEHAVIOR (as in streamlit)")
    print("=" * 60)
    
    # Test multiple random vectors to see the range of predictions
    random_predictions = []
    for i in range(10):
        random_features = np.random.rand(1, 81)  # Random features as in the fallback
        pred = model.predict(random_features)[0]
        prob = model.predict_proba(random_features)[0]
        random_predictions.append((pred, prob))
        
        print(f"Random test {i+1}: Prediction={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Count how many predict high vs low burnout
    high_burnout_count = sum(1 for pred, _ in random_predictions if pred == 1)
    print(f"\nOut of 10 random tests: {high_burnout_count} predicted High Burnout, {10-high_burnout_count} predicted Low Burnout")
    return random_predictions

if __name__ == "__main__":
    print("🔍 TESTING EXTREME CASES AND FALLBACK BEHAVIOR")
    print()
    
    # Load model and data
    model, df = load_model_and_data()
    sample_df = df.sample(n=1000, random_state=42)
    sample_df = add_derived_features(sample_df)
    
    # Create proper preprocessor
    preprocessor, feature_columns = create_proper_preprocessor(sample_df)
    
    # Test with extreme variations
    print("=" * 60)
    print("TESTING EXTREME VARIATIONS")
    print("=" * 60)
    
    base_case = {
        'Age': 35,
        'Gender': 'Male',
        'Occupation': 'IT',
        'Country': 'USA',
        'Mental_Health_Condition': 'No',
        'Severity': np.nan,
        'Consultation_History': 'No',
        'Sleep_Hours': 7,
        'Work_Hours': 40,
        'Diet_Quality': 'Average',
        'Smoking_Habit': 'Non-Smoker',
        'Alcohol_Consumption': 'Social Drinker',
        'Medication_Usage': 'No'
    }
    
    # Test cases exactly as user described
    case1 = base_case.copy()
    case1.update({'Physical_Activity_Hours': 0, 'Social_Media_Usage': 0.5})  # Min SM value
    
    case2 = base_case.copy() 
    case2.update({'Physical_Activity_Hours': 0, 'Social_Media_Usage': 1.0})  # Slightly > 0
    
    case3 = base_case.copy()
    case3.update({'Physical_Activity_Hours': 0, 'Social_Media_Usage': 3.0})  # More > 0
    
    # Also test truly 0 social media
    case0 = base_case.copy()
    case0.update({'Physical_Activity_Hours': 0, 'Social_Media_Usage': 0.1})  # As close to 0 as possible
    
    test_case("Case 0 (PA=0, SM≈0)", case0, model, preprocessor, feature_columns)
    test_case("Case 1 (PA=0, SM=0.5)", case1, model, preprocessor, feature_columns)
    test_case("Case 2 (PA=0, SM=1.0)", case2, model, preprocessor, feature_columns)  
    test_case("Case 3 (PA=0, SM=3.0)", case3, model, preprocessor, feature_columns)
    
    # Test the random fallback behavior
    test_random_fallback_behavior(model)