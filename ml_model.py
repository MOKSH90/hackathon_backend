import joblib
from threading import Lock

# Path to your saved model file
MODEL_PATH = "triage_model.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Thread lock to ensure safe concurrent access
model_lock = Lock()

