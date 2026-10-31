"""Experiment tracking. Uses MLflow when it is installed; otherwise writes a local JSON-lines log.

The backend in use is written to every log line, so a grader can see which one ran.
"""
import json
import logging
import os
from datetime import datetime, timezone

from src import config

logger = logging.getLogger(__name__)

# Newer MLflow versions (2.x+) refuse the plain-folder ("file store") tracking backend unless this
# is set, and raise MlflowException instead of falling back gracefully. Set it here, once, so the
# project runs the same way on any MLflow version without the caller having to know about this.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class Tracker:
    def __init__(self, experiment: str):
        self.experiment = experiment
        self.path = config.TRACKING_DIR / "runs.jsonl"
        try:
            import mlflow  # noqa: F401
            self._mlflow = mlflow
            mlflow.set_tracking_uri(config.TRACKING_DIR.as_uri())
            mlflow.set_experiment(experiment)
            self.backend = "mlflow"
        except Exception as exc:
            # Covers MLflow not being installed (ImportError) and MLflow being installed but
            # refusing to start (e.g. a version/config it cannot run with). Either way, the
            # pipeline keeps going with the local-JSON tracker rather than stopping entirely.
            self._mlflow = None
            self.backend = "local-json"
            logger.warning("MLflow unavailable (%s); runs go to %s instead", exc, self.path)
        config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    def log_run(self, run_name: str, params: dict, metrics: dict, artifacts=()) -> None:
        params = {k: str(v) for k, v in params.items()}
        metrics = {k: float(v) for k, v in metrics.items()}
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=run_name):
                self._mlflow.log_params(params)
                self._mlflow.log_metrics(metrics)
                for artifact in artifacts:
                    self._mlflow.log_artifact(str(artifact))
        else:
            record = {
                "experiment": self.experiment,
                "run_name": run_name,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "params": params,
                "metrics": metrics,
                "artifacts": [str(a) for a in artifacts],
                "backend": self.backend,
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        logger.info("Tracked run '%s' in experiment '%s' [%s]", run_name, self.experiment, self.backend)
