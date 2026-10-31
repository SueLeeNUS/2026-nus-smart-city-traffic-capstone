"""Run the whole Part 3 pipeline end to end, logging to the console and to logs/pipeline.log.

    python run_all.py

Deep learning runs only if PyTorch is installed; otherwise it is skipped and the skip is logged.
"""
import logging
import sys

from src import config, data_prep, features, mlops, responsible_ai, supervised, unsupervised


def configure_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(config.LOG_DIR / "pipeline.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = [file_handler, console]


def main() -> int:
    configure_logging()
    log = logging.getLogger("run_all")
    try:
        clean = data_prep.load_clean()  # Part 2's cleaned file, unchanged
        frame = features.build_frame(clean)
        supervised.run(frame)
        unsupervised.run(frame)
        responsible_ai.run(frame)
        mlops.run(frame)
        try:
            from src import deep_learning
            deep_learning.run()
        except ImportError as exc:
            log.warning("Deep learning skipped (%s). Install torch and shap, then run `python -m src.deep_learning`.", exc)
    except FileNotFoundError:
        log.error("Pipeline stopped: raw data file missing at %s", config.RAW_PATH)
        return 1
    except Exception:
        log.exception("Pipeline stopped with an unexpected error")
        return 1
    log.info("Pipeline finished. Log written to %s", config.LOG_DIR / "pipeline.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
