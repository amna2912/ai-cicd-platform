import pickle
import pandas as pd
from typing import Dict, List
from app.training.train import explain_single_prediction  

class PredictionService:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.explainer = None
        self.feature_names = []
        self.load_models()

    def load_models(self):
        try:
            with open(f"{self.model_dir}/failure_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            
            with open(f"{self.model_dir}/shap_explainer.pkl", "rb") as f:
                self.explainer = pickle.load(f)
            
            import json
            with open(f"{self.model_dir}/model_meta.json") as f:
                meta = json.load(f)
                self.feature_names = meta["feature_names"]
            
            print("✅ Modèles ML chargés avec succès")
            return True
        except Exception as e:
            print(f"⚠️ Erreur chargement modèles: {e}")
            return False

    def predict(self, features: Dict) -> Dict:
        if not self.model:
            return {"failure_probability": 0.5, "risk_level": "UNKNOWN", "explanation": {}}

        feature_vector = {name: features.get(name, 0.0) for name in self.feature_names}
        X = pd.DataFrame([feature_vector])

        prob = float(self.model.predict_proba(X)[0][1])

        risk_level = (
            "CRITICAL" if prob > 0.75 else
            "HIGH" if prob > 0.55 else
            "MEDIUM" if prob > 0.30 else "LOW"
        )

        explanation = {}
        if self.explainer:
            try:
                explanation = explain_single_prediction(self.explainer, X, self.feature_names)
            except:
                pass

        return {
            "failure_probability": round(prob, 4),
            "risk_level": risk_level,
            "explanation": explanation,
            "features": feature_vector
        }