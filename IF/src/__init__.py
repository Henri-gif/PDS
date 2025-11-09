# Map 'from src import models' to Isolation Forest models for convenience
try:
    import src.models_iforest as models
except Exception:
    from . import models_iforest as models

build_model = models.build_model
fit_model = models.fit_model
score_samples = models.score_samples
save_model = models.save_model
load_model = models.load_model
