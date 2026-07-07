
"""evaluate.py - metric persistence + reporting."""
import json
from pathlib import Path
def save_metrics(m,path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    open(path,"w").write(json.dumps(m,indent=2)); return m
def print_report(m):
    print("="*44); print("Evaluation report"); print("="*44)
    for k,v in m.items(): print(("  %-22s: %.4f"%(k,v)) if isinstance(v,float) else ("  %-22s: %s"%(k,v)))
