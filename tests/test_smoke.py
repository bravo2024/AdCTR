
from src.data import make_synthetic
from src.model import fit_and_evaluate
def test_pipeline_runs():
    model,metrics=fit_and_evaluate(make_synthetic())
    assert isinstance(metrics,dict) and len(metrics)>=2
    assert model is not None
