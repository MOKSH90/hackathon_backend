import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np
import joblib
df = pd.read_csv("balanced_triage_20.csv")

df = df.dropna()
print(df.columns)
if "condition" in df.columns:
    df = df.drop(columns=["condition"])
X = df.drop(["severity_score"], axis=1)
y = df["severity_score"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state = 42)

param_grid = {
    "n_estimators": [300,500],
    "max_depth": [20,None],
    "min_samples_split": [2,5],
    "max_features": ["sqrt", "log2"]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE:{mse: .2f} ")
print(f"Test R^2: {r2: .2f}")

feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
print("Feature Importances\n", feature_importance.sort_values(ascending=False))
joblib.dump(best_model , "triage_model.pkl") 
joblib.dump(scaler, "scaler.pkl")
print("Both the models hae been trained and saved successfully")
print("Features during training:")
print(X.columns.tolist())


# # Confidence Calculation
# # -------------------
# confidence_threshold = 0.5  # between 0 and 1

# # Predictions from each tree
# all_tree_preds = np.array([tree.predict(X_test) for tree in best_model.estimators_])

# # Standard deviation of predictions
# prediction_std = np.std(all_tree_preds, axis=0)

# # Confidence score: higher std → lower confidence
# confidence_scores = 1 / (1 + prediction_std)

# # Generate confidence messages
# confidence_messages = [
#     "I am confident" if conf >= confidence_threshold else "I am not confident"
#     for conf in confidence_scores
# ]

# # Combine results
# results_df = pd.DataFrame({
#     "predicted_severity": y_pred,
#     "confidence_score": confidence_scores,
#     "confidence_message": confidence_messages
# })

# print(results_df.head())
