# Drug Side Effect Severity Prediction 🚑⚡

A machine learning project that predicts side effect severity of drugs based on dosage, expiry date, and categorical features (drug type, manufacturer, etc.). Built with Random Forest, XGBoost, and Gradient Boosting pipelines, deployed with FastAPI for real-time predictions.

## 🔥 What This Project Is

The goal: predict the severity of drug side effects (Mild / Moderate / Severe) given input information such as dosage, drug class, indications, contraindications, warnings, etc.

This repo contains:

- 📓 Data Preprocessing & Model Training Notebook → complete ML pipeline

- ⚙️ Saved Best Pipeline (best_pipeline.joblib)

- 🌐 FastAPI Backend for real-time inference

## 🗂 Repository Structure

project_root/
│
├─ notebooks/
│   ├─ BaseModel(Random_Forest_Classifier).ipynb
│   ├─ Best_Model.ipynb
│   ├─ Comparison Model.ipynb
│   ├─ Cross_Validation.ipynb
│   ├─ Drug Labels and Side Effects Analysis.ipynb
│   ├─ EDA.ipynb
│   ├─ Gradient_Bosting.ipynb
│   ├─ Inference.ipynb
│   ├─ Logistic_Regression.ipynb
│   ├─ Preprocessing.ipynb
│   ├─ Random_Forest_Classifier(FeatureSelection).ipynb
│   ├─ Randomized_Search_CV.ipynb
│   ├─ XgBoost.ipynb
│
├─ source/model/
│   ├─ Best_Model.ipynb
│   ├─ best_pipeline.joblib
│   ├─ label_encoder.joblib
│   ├─ app.py
│   ├─ inference.py
│
├─ drug_side_effect.csv
├─ requirements.txt
├─ .gitignore
├─ README.md




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

### 6. Best-Model(Gradient-Boosting).ipynb

- Chooses Gradient Boosting as the best performing model.
  
- Retrains it on the full training dataset. 

- Saves the trained model using joblib for future inference. 

### 7. Infererence-best-model.ipynb

- Loads the saved Gradient Boosting model
  
- Demonstrates how to make predictions on new/unseen data
  
- Explains each step of inference clearly

## 📊 Model Performance & Comparison
During development, I experimented with several models to predict drug side effect severity and evaluated them on accuracy, log loss, and class-level metrics to determine the best approach.

| Model                                              | Test Accuracy | Test Log Loss | Notes                                                                                                                    |
| -------------------------------------------------- | ------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Gradient Boosting (Final)**                      | 0.378         | 1.18          | Slightly better accuracy than other models; relatively balanced across classes but overall predictions are weak.         |
| **Random Forest (After Feature Selection)**        | 0.32          | 1.180         | Feature selection decreased performance. Model overfits moderately; poor generalization on test data.                    |
| **Gradient Boosting (CV & Hyperparameter Tuning)** | 0.33–0.36     | 1.10–1.11     | High overfitting (train 0.92 vs test \~0.33). Performs well on class 0 but fails on class 2.                             |
| **XGBoost**                                        | 0.3576        | 1.098         | Lowest log loss but heavily biased toward class 0. Underfits on training data; predictions skewed toward majority class. |
| **Logistic Regression**                            | 0.347         | 1.180         | Baseline linear model; underfits, uniform predictions across classes, highlighting dataset complexity.               |

### 🔍 Key Insights

- Best Accuracy: Gradient Boosting (Final)

- Best Log Loss: XGBoost, but suffers from severe class imbalance

- Overfitting Analysis:

  - GB (CV/Tuned) → extreme overfit (train 0.92 vs test ~0.33)

  - RF → moderate overfit

  - XGBoost → slight underfit but biased predictions

- Class Imbalance & Bias:

  - XGBoost favors class 0 heavily

  - GB and RF more balanced but low overall performance

  - Logistic Regression uniform but underfit

### Conclusion:
Gradient Boosting provided the best trade-off between accuracy and class balance. XGBoost minimized log loss but failed to generalize for minority classes. Logistic Regression served as a baseline, emphasizing the need for non-linear models to handle complex relationships in the dataset.

## 🛠️ Tech Stack

- Languages: Python

- Libraries: Pandas, Scikit-learn, Matplotlib, XGBoost 

- Deployment: FastAPI, Swagger UI

- Model Persistence: Joblib

---

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
  
- Learned feature selection (embedded methods, correlation, importance).

- Build production-ready ML pipelines (not just experiments in a notebook).

- Practiced hyperparameter tuning with stratified K-Fold.

- Deploy models with FastAPI so they’re accessible as a real-world service.

- Implement robust error handling and input validation in APIs.

## 📈 Visualizations

- Confusion matrix heatmaps for model evaluation.
  
- Feature importance plots.
  
- Correlation heatmaps.
  
- Comparison of all models.

## 🚀 Future Plans

- Containerize this app with Docker for easy deployment.

- Add a frontend UI for non-technical users.

- Scaling FastAPI deployment with Docker & Kubernetes.


## 📬 Connect With Me

LinkedIn: https://www.linkedin.com/in/rose-kc-0622ba315/

Email: kcr3307@gmail.com

### ⚠️ Disclaimer: This project is for educational purposes only. Predictions should not be used for actual medical decision-making.








