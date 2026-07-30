"""
W&B Experiment Tracking für Spam-Klassifikation
===============================================
Integriert Weights & Biases in die Spam-Klassifikation.
Loggt Modell-Ergebnisse, Feature-Statistiken und Top-Features.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="spam-klassifikation", config={...})
    tracker.log_model_result("NaiveBayes", accuracy=0.95, f1=0.93)
    tracker.finish()
"""

import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für Spam-Klassifikation.

    Features:
    - Modell-Ergebnisse (Accuracy, F1, Precision, Recall)
    - Feature-Statistiken
    - Top-Features visualisieren
    """

    def __init__(self, project: str = "spam-klassifikation",
                 config: dict = None, tags: list = None,
                 group: str = None, job_type: str = "train",
                 notes: str = None, offline: bool = False):
        self.project = project
        self.run = None

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["spam", "nlp"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_model_result(self, model_name: str, accuracy: float = None,
                         f1: float = None, precision: float = None,
                         recall: float = None, train_time: float = None):
        """Loggt Ergebnisse eines Modells."""
        metrics = {}
        if accuracy is not None:
            metrics[f"model/{model_name}/accuracy"] = accuracy
        if f1 is not None:
            metrics[f"model/{model_name}/f1"] = f1
        if precision is not None:
            metrics[f"model/{model_name}/precision"] = precision
        if recall is not None:
            metrics[f"model/{model_name}/recall"] = recall
        if train_time is not None:
            metrics[f"model/{model_name}/train_time"] = train_time
        self.log(metrics)

    def log_feature_stats(self, num_features: int, num_samples: int,
                          spam_ratio: float):
        """Loggt Feature-Statistiken."""
        self.log({
            "data/num_features": num_features,
            "data/num_samples": num_samples,
            "data/spam_ratio": spam_ratio,
        })

    def log_top_features(self, model_name: str, features: list,
                         weights: list):
        """Loggt die wichtigsten Features eines Modells."""
        if not self.run:
            return
        table = wandb.Table(columns=["feature", "weight"])
        for f, w in zip(features, weights):
            table.add_data(f, w)
        self.run.log({f"model/{model_name}/top_features": table})

    def finish(self):
        """Beendet den W&B-Run. Sicher bei mehrfachem Aufruf."""
        if self.run:
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
