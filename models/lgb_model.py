import pandas as pd
import lightgbm as lgb  # ✅ Correct import for LightGBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

DATA_FILE = "../data/training_data.csv"
MODEL_SAVE_FILE = "../models/lgb_model.pkl"

df = pd.read_csv(DATA_FILE)

features = ['time_since_last_access', 'frequency_count']
label = 'time_to_next_request'

X = df[features]
y = df[label]

print(f"Data loaded. Found {len(X)} samples.")
print("Features:", features)
print("Label:", label)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

lgbm = lgb.LGBMRegressor(  # ✅ Use LGBMRegressor
    objective='regression_l1', 
    n_estimators=1000,         
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1  # Suppress LightGBM warnings
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

print("\nEvaluating model performance...")
predictions = lgbm.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, predictions))
print(f"Validation RMSE: {rmse:.4f}")
print("This means, on average, the model's prediction for the time to the next request is off by ~{:.2f} timestamps.".format(rmse))


# --- 6. Save the Trained Model 💾 ---
print(f"\nSaving the trained model to '{MODEL_SAVE_FILE}'...")
joblib.dump(lgbm, MODEL_SAVE_FILE)
print("Model saved successfully!")