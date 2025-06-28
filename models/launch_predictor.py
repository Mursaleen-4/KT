import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SimplePredictor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.rocket_weights = {
            'Falcon 9': 0.9,
            'Falcon Heavy': 0.85
        }
        self.weather_weights = {
            'Clear': 1.0,
            'Cloudy': 0.9,
            'Rainy': 0.7,
            'Windy': 0.8
        }
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return np.ones(len(X))
    
    def predict_proba(self, X):
        probs = []
        for _, row in X.iterrows():
            # Simple weighted average of factors
            rocket_weight = self.rocket_weights.get(row['rocket_name'], 0.8)
            weather_weight = self.weather_weights.get(row['weather_condition'], 0.8)
            wind_factor = max(0, 1 - (row['wind_speed'] / 100))
            temp_factor = 1.0 if 15 <= row['temperature'] <= 30 else 0.8
            
            success_prob = (rocket_weight + weather_weight + wind_factor + temp_factor) / 4
            probs.append([1 - success_prob, success_prob])
        
        return np.array(probs) 