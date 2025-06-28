from models.launch_predictor import SimplePredictor
import joblib

# Create and save the model
model = SimplePredictor()
joblib.dump(model, 'models/launch_predictor.joblib') 