# Mental Health & Burnout Prediction Demo

A Streamlit web application that demonstrates machine learning-based burnout prediction using mental health and lifestyle data.

## Features

- 🏠 **Overview**: Introduction to the model and dataset statistics
- 📊 **Dataset Explorer**: Interactive exploration of the 50,000+ mental health records
- 🎯 **Burnout Prediction**: Real-time prediction using trained Random Forest model
- 📈 **Model Insights**: Feature importance and performance metrics

## Quick Start

1. **Install Requirements**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run the Demo**
   ```bash
   streamlit run mental_health_demo.py
   ```

3. **Open in Browser**
   Navigate to: `http://localhost:8501`

## Files

- `mental_health_demo.py` - Main Streamlit application
- `app.py` - Original demo app with EDA focus
- `dashboard.py` - Simple dashboard version
- `trained_model.pkl` - Pre-trained Random Forest model
- `requirements.txt` - Python dependencies

## Model Details

- **Algorithm**: Random Forest Classifier
- **Performance**: 77.1% accuracy, 84.5% precision
- **Features**: Demographics, lifestyle factors, work patterns, health indicators
- **Target**: High stress level (burnout proxy)

## Dataset

The model uses a mental health workplace survey dataset with 50,000 records containing:
- Demographics (age, gender, occupation, country)
- Lifestyle factors (sleep, exercise, diet, habits)
- Work patterns (hours, stress levels)
- Mental health indicators

## Usage

1. Navigate through different sections using the sidebar
2. Explore dataset statistics and visualizations
3. Use the prediction form to assess burnout risk
4. View model performance and feature importance

## Prediction Form

Enter personal information including:
- Demographics (age, gender, occupation, country)
- Work patterns (hours per week)
- Lifestyle (sleep hours, exercise, diet, social media usage)
- Health factors (mental health status, consultation history, medication)
- Habits (smoking, alcohol consumption)

The model will predict burnout risk and provide risk factor analysis.

## Technical Details

- **Feature Engineering**: Automatically creates bracket columns (age groups, work hour ranges, etc.) as used in training
- **Preprocessing**: Replicates the exact 81-feature pipeline from the notebook
- **Categorical Handling**: Proper one-hot encoding for all categorical variables
- **Missing Values**: Handles missing data using median (numeric) and mode (categorical) imputation