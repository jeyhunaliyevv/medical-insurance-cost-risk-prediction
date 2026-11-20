"""Predict high-cost probability for a validated example (local trusted model only)."""
from pathlib import Path
import joblib
import pandas as pd
from .pipeline import engineer

def predict_from_rows(model, rows):
    x = engineer(pd.DataFrame(rows))
    prob = model.predict_proba(x)[:,1]
    return [{"high_cost_probability": round(float(p), 4),
             "high_cost_prediction_at_0_5": int(p >= .5)} for p in prob]

def main():
    path = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
    if not path.exists():
        raise SystemExit("Train first: python -m src.train")
    # Only load artifacts you generated yourself: joblib uses pickle internally.
    model = joblib.load(path)
    example = {"age": 35, "sex": "female", "bmi": 27.1, "children": 1,
               "smoker": "no", "region": "northeast"}
    print(predict_from_rows(model,[example]))

if __name__ == "__main__": main()
