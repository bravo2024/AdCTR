
"""train.py - build (synthetic) data, train, evaluate, persist. Runs with no downloads."""
from src.data import make_synthetic
from src.model import fit_and_evaluate
from src.evaluate import save_metrics, print_report
from src.persist import save_model
def main():
    data=make_synthetic()
    model,metrics=fit_and_evaluate(data)
    save_model(model); save_metrics(metrics); print_report(metrics)
    print("\nSaved model -> models/model.pkl and metrics -> models/metrics.json")
if __name__=="__main__": main()
