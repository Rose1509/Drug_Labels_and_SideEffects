# Drug Side Effect Severity Prediction 🚑⚡


## 🔥 What This Project Is

The goal: predict the severity of drug side effects (Mild / Moderate / Severe) given input information such as dosage, drug class, indications, contraindications, warnings, etc.

This repo contains:

- 📓 Data Preprocessing & Model Training Notebook → complete ML pipeline

- ⚙️ Saved Best Pipeline (best_pipeline.joblib)

- 🌐 FastAPI Backend for real-time inference



## 🧩 Project Breakdown
### 1. Data Preprocessing & Cleaning🧹

- Removed duplicates and handled missing values.

- Detected and treated outliers in dosage and expiry-related features.

- Converted dates (expiry_date) into a numeric feature: days_until_expiry.

### 2. Feature Engineering🛠️

- Encoded categorical features with OrdinalEncoder and OneHotEncoder.

- Created new engineered features to better capture relationships.

- Balanced the dataset with SMOTE to fix class imbalance (important for side effect severity).

### 3. Model Development & Training🤖

I compared multiple models to see what worked best:

- ✅ Random Forest Classifier

- ✅ XGBoost Classifier

- ✅ Gradient Boosting Classifier
  
- ✅ Cross Validation


For each model:

- Tuned hyperparameters using RandomizedSearchCV.

- Evaluated with multiple metrics: log loss, balanced accuracy, macro F1 score, and confusion matrices.

- Plotted feature importance for explainability.

### 4. Best Model Selection🏆

- The best performing pipeline was exported as best_pipeline.joblib for production use.

- Final model can predict severity classes:

  - 0 → 😌 Mild

  - 1 → 😐 Moderate

  - 2 → 😫 Severe

### 5. FastAPI Deployment⚡

I built a REST API using FastAPI that loads the trained pipeline and makes predictions:

- Lifespan event ensures the model is loaded once at startup.

- /predict endpoint accepts drug details (dosage, class, side effects, warnings, etc.) and returns predicted severity.

- Automatic date handling: expiry_date is converted into days_until_expiry before feeding into the pipeline.

- Error handling: If model fails to load or input is invalid, the API returns a clear JSON error response.

📌**Example request:**
POST /predict
{
  "dosage_mg": 500,
  "drug_class": "Antibiotic",
  "indications": "Bacterial infection",
  "side_effects": "Nausea",
  "contraindications": "Liver disease",
  "warnings": "Avoid alcohol",
  "approval_status": "Approved",
  "expiry_date": "2026-05-10"
}

📌**Example response:**
{
  "status_code": 200,
  "prediction_class": "Moderate"
}

 

## 📚 Key Learning Outcomes

Through this project, I’ve learned to:

- Handle messy medical/pharma datasets with systematic preprocessing.

- Use SMOTE properly to balance classes without leaking test data.

- Build production-ready ML pipelines (not just experiments in a notebook).

- Deploy models with FastAPI so they’re accessible as a real-world service.

- Implement robust error handling and input validation in APIs.

 

## 🚀 Future Plans

- Containerize this app with Docker for easy deployment.

- Add a frontend UI for non-technical users.

- Deploy on AWS/GCP/Heroku for live access.



## 📬 Connect With Me

LinkedIn: https://www.linkedin.com/in/rose-kc-0622ba315/

Email: kcr3307@gmail.com



### ⚠️ Disclaimer: This project is for educational purposes only. Predictions should not be used for actual medical decision-making.







