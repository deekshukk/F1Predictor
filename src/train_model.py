from feature import create_features_and_target
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load features and target
features, target = create_features_and_target()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
print("Training Random Forest model...")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/random_forest_f1_model.pkl")
print(" Model saved to model/random_forest_f1_model.pkl")
