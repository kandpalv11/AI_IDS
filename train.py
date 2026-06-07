import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading dataset...")

# Dataset Load
df = pd.read_csv("dataset/DrDoS_DNS.csv")

# Missing values remove
df = df.dropna()

# Label column
target_column = "label"

# Features & Target
X = df[[
    "protocol",
    "flow_duration",
    "total_forward_packets",
    "total_backward_packets"
]]

y = df["label"]
# Convert text columns to numbers
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Model...")

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "ids_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel Saved Successfully!")
print("Files Created:")
print(" - ids_model.pkl")
print(" - label_encoder.pkl")