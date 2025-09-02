import os
import pandas as pd
import joblib


class DrugSideEffectPredictor:
    def __init__(self, model_file="best_pipeline.joblib", encoder_file="label_encoder.joblib"):
        """
        Initialize the predictor by loading the trained pipeline and label encoder.
        Paths are relative to the script location.
        """
        # Compute absolute paths relative to this script
        base_dir = os.path.join(os.path.dirname(__file__), "../model")
        model_path = os.path.join(base_dir, model_file)
        encoder_path = os.path.join(base_dir, encoder_file)

        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print("✅ Model and encoder loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model/encoder from {base_dir}: {e}")

    def get_user_input(self):
        """
        Collect user input for a new drug, compute days_until_expiry,
        and return a DataFrame ready for prediction.
        """
        print("\nPlease enter the following information to predict side effect severity:\n")
        user_data = {}

        try:
            # Numeric features
            user_data['dosage_mg'] = float(input("Dosage (mg): "))

            # Categorical features
            user_data['drug_class'] = input("Drug Class: ")
            user_data['indications'] = input("Indications: ")
            user_data['side_effects'] = input("Side Effects: ")
            user_data['contraindications'] = input("Contraindications: ")
            user_data['warnings'] = input("Warnings: ")
            user_data['approval_status'] = input("Approval Status: ")

            # Expiry date
            expiry_input = input("Expiry date (YYYY-MM-DD): ")
            expiry_dt = pd.to_datetime(expiry_input)
            user_data['days_until_expiry'] = (expiry_dt - pd.Timestamp('today')).days
            user_data['expiry_date'] = user_data['days_until_expiry']  # consistent with training

        except ValueError:
            print("\n❌ Invalid input. Make sure numeric fields are numbers and date is YYYY-MM-DD.")
            return None

        return pd.DataFrame([user_data])

    def predict(self, input_df: pd.DataFrame):
        """
        Predict side effect severity given a DataFrame.
        """
        try:
            pred_encoded = self.model.predict(input_df)
            pred_label = self.label_encoder.inverse_transform(pred_encoded)
            return pred_label[0]
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


if __name__ == "__main__":
    # Create predictor instance
    predictor = DrugSideEffectPredictor()

    # Collect input from CLI
    input_df = predictor.get_user_input()

    if input_df is not None:
        result = predictor.predict(input_df)
        print("\n--- Prediction Result ---")
        print(f"The predicted side effect severity is: {result}")
