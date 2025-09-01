import pandas as pd
import joblib

# === Load saved model and encoder ===
best_pipeline = joblib.load("best_pipeline.joblib")
le = joblib.load("label_encoder.joblib")

# Load saved model
model_pipeline = joblib.load("best_pipeline.joblib")
print("Loaded trained model")

# === User Input Section ===
def get_user_drug_input():
    """
    Collects user input for a new drug, computes days_until_expiry from expiry_date,
    and returns a dictionary inside a list ready for prediction.
    """
    print("Please enter the following information to predict side effect severity:")

    user_data = {}
    try:
        # Numeric features (except days_until_expiry)
        user_data['dosage_mg'] = float(input("Dosage (mg): "))

        # Categorical features
        user_data['drug_class'] = input("Drug Class: ")
        user_data['indications'] = input("Indications: ")
        user_data['side_effects'] = input("Side Effects: ")
        user_data['contraindications'] = input("Contraindications: ")
        user_data['warnings'] = input("Warnings: ")
        user_data['approval_status'] = input("Approval Status: ")

        # Expiry date input
        expiry_input = input("Expiry date (YYYY-MM-DD): ")
        expiry_dt = pd.to_datetime(expiry_input)
        user_data['days_until_expiry'] = (expiry_dt - pd.Timestamp('today')).days
        user_data['expiry_date'] = user_data['days_until_expiry']  # keep consistent with training

    except ValueError:
        print("\nInvalid input. Please make sure numeric fields are numbers and date format is YYYY-MM-DD.")
        return None

    return [user_data]  # list of dicts for DataFrame


# === Get user input and make prediction ===
user_input_data = get_user_drug_input()

if user_input_data:
    new_df = pd.DataFrame(user_input_data)

    # Predict using trained pipeline
    pred_encoded = best_pipeline.predict(new_df)
    pred_label = le.inverse_transform(pred_encoded)

    print("\n--- Prediction Result ---")
    print(f"The predicted side effect severity is: {pred_label[0]}")

