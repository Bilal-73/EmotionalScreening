# neuroticure_model_training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =========================
# 1️⃣ Load Dataset
# =========================
df = pd.read_csv("BackEnd/neuroticure_training_data.csv")

print("\n🔍 Dataset Preview:")
print(df.head())

# # =========================
# # 2️⃣ Basic Analysis
# # =========================

# # Age distribution
# plt.figure(figsize=(6, 4))
# sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")
# plt.title("Age Distribution")
# plt.show()

# # Gender distribution
# plt.figure(figsize=(5, 4))
# sns.countplot(x="Gender", data=df, palette="pastel")
# plt.title("Gender Distribution")
# plt.show()

# # Redirection category distribution
# plt.figure(figsize=(6, 4))
# sns.countplot(x="Redirection", data=df, palette="muted")
# plt.title("Redirection Category Distribution")
# plt.xticks(rotation=30)
# plt.show()

# =========================
# 3️⃣ Preprocessing
# =========================

# Encode categorical features
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

target_encoder = LabelEncoder()
df["Redirection"] = target_encoder.fit_transform(df["Redirection"])

# Features & Target
X = df[["Age", "Gender", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]]
y = df["Redirection"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =========================
# 4️⃣ Train Model
# =========================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =========================
# 5️⃣ Evaluation
# =========================
y_pred = model.predict(X_test)

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
# plt.title("Confusion Matrix")
# plt.ylabel("Actual")
# plt.xlabel("Predicted")
# plt.show()

# # =========================
# 6️⃣ Save Model & Encoders
# =========================
joblib.dump(model, "neuroticure_model.pkl")
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("\n✅ Model and encoders saved successfully!")
