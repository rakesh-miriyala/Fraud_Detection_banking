# train_model.py

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Create necessary folders
os.makedirs('./model', exist_ok=True)
os.makedirs('./visuals', exist_ok=True)

# Load dataset
df = pd.read_csv('./data/creditcard.csv')

# Sample (optional for dev speed)
df = df.sample(n=30000, random_state=42)

# Prepare features and label
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Scale only 'Amount'
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

# Save feature list for batch inference compatibility
features_used = X.columns.tolist()
joblib.dump(features_used, './model/feature_list.pkl')

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model training
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, './model/fraud_model.pkl')
joblib.dump(scaler, './model/scaler.pkl')
print("✅ Model and scaler saved.")

# 📊 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("./visuals/confusion_matrix.png")
plt.close()

# 📈 ROC Curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='XGBoost (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("./visuals/roc_curve.png")
plt.close()

# 🧠 SHAP Explainability
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('./visuals/shap_summary_plot.png')
plt.close()

# Force plot for a single prediction
example = X_test.iloc[[0]]
force_plot = shap.force_plot(explainer.expected_value, shap_values[0].values, feature_names=X_test.columns)

# Save as interactive HTML
shap.save_html('./visuals/shap_force_plot.html', force_plot)


print("✅ Visualizations (confusion matrix, ROC, SHAP) saved.")
