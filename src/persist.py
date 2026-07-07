
"""persist.py - model save/load via stdlib pickle."""
import pickle
from pathlib import Path
def save_model(model,path="models/model.pkl"):
    Path(path).parent.mkdir(parents=True,exist_ok=True); pickle.dump(model,open(path,"wb")); return path
def load_model(path="models/model.pkl"): return pickle.load(open(path,"rb"))
