"""
W&B Experiment Tracking für Spam-Klassifikation
===============================================
Integriert Weights & Biases in die Spam-Klassifikation.
Loggt Modellvergleiche, Metriken und Feature-Analyse.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="spam-klassifikation", config={...})
    tracker.log_model_result("NaiveBayes", accuracy=0.95, f1=0.93)
    tracker.finish()
"""

import os
import subprocess
import time
from typing import Optional, List

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für Spam-Klassifikation.

    Features:
    - Modellvergleich (Naive Bayes, Logistic Regression, NN)
    - Metrik-Logging (Accuracy, F1, Precision, Recall)
    - Feature-Analyse (TF-IDF Dimensionen)
    - Trainingszeit-Messung
    """

    def __init__(
        self,
        project: str = "spam-klassifikation",
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        notes: Optional[str] = None,
        offline: bool = False,
    ):
        self.project = project
        self.run = None
        self._start_time = time.time()

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["spam", "nlp", "classification"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL,
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: Optional[int] = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_model_result(
        self,
        model_name: str,
        accuracy: float,
        f1: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        train_time: float = 0.0,
        params: Optional[dict] = None,
    ):
        """Loggt Ergebnisse eines Modells."""
        metrics = {
            f"{model_name}/accuracy": accuracy,
            f"{model_name}/f1_score": f1,
            f"{model_name}/precision": precision,
            f"{model_name}/recall": recall,
            f"{model_name}/train_time": train_time,
        }
        if params:
            for k, v in params.items():
                metrics[f"{model_name}/param_{k}"] = v
        self.log(metrics)

    def log_feature_stats(self, num_features: int, num_samples: int,
                          spam_ratio: float):
        """Loggt Feature- und Datensatz-Statistiken."""
        self.log({
            "data/num_features": num_features,
            "data/num_samples": num_samples,
            "data/spam_ratio": spam_ratio,
        })

    def log_top_features(self, model_name: str, features: List[str],
                         weights: List[float]):
        """Loggt die wichtigsten Features eines Modells."""
        if not self.run:
            return
        table = wandb.Table(columns=["feature", "weight"])
        for f, w in zip(features[:20], weights[:20]):
            table.add_data(f, w)
        self.run.log({f"{model_name}/top_features": table})

    def finish(self):
        """Beendet den W&B-Run."""
        elapsed = time.time() - self._start_time
        if self.run:
            self.log({"total_time_seconds": elapsed})
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
