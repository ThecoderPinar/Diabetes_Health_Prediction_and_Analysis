from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("./models/pipeline_xgboost.pkl")

# Load the feature names
X_train = pd.read_csv("../data/processed/X_train_engineered.csv")
feature_names = X_train.columns.tolist()

# Remove 'PatientID' from feature names if it exists
if 'PatientID' in feature_names:
    feature_names.remove('PatientID')

# Categorize features
demographic_features = [feature for feature in feature_names if
                        'age' in feature or 'gender' in feature or 'ethnicity' in feature]
medical_history_features = [feature for feature in feature_names if 'history' in feature or 'diabetes' in feature]
lifestyle_features = [feature for feature in feature_names if
                      'bmi' in feature or 'smoking' in feature or 'activity' in feature]

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', demographic_features=demographic_features,
                           medical_history_features=medical_history_features,
                           lifestyle_features=lifestyle_features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        features = [float(request.form[feature]) for feature in feature_names]
        input_data = pd.DataFrame([features], columns=feature_names)

        # Predict using the model
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0][1]

        return render_template('index.html', prediction=prediction, probability=prediction_prob,
                               demographic_features=demographic_features,
                               medical_history_features=medical_history_features,
                               lifestyle_features=lifestyle_features)
    except KeyError as e:
        return f"Missing form data for feature: {e.args[0]}", 400
    except ValueError as e:
        return str(e), 400


if __name__ == "__main__":
    app.run(debug=True)
