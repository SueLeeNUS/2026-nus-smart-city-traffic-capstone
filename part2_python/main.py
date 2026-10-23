"""Run the whole project end to end: clean -> features -> figures.

    python main.py            # normal run (INFO and above)
    python main.py --debug    # also show DEBUG messages
"""

import argparse
import logging
import sys
from pathlib import Path

import feature_engineering
import pipeline
import visualizations
from logging_config import setup_logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the traffic analytics pipeline end to end.")
    parser.add_argument("--debug", action="store_true", help="show DEBUG messages")
    parser.add_argument("--log-file", default=str(BASE_DIR / "pipeline.log"))
    parser.add_argument("--raw-path", default=str(pipeline.RAW_PATH), help="raw CSV to process")
    parser.add_argument("--append-log", action="store_true", help="add to the log file instead of replacing it")
    args = parser.parse_args()
    setup_logging(args.log_file, debug=args.debug, file_mode="a" if args.append_log else "w")

    try:
        pipeline.run_pipeline(Path(args.raw_path))
        feature_engineering.run_feature_engineering()
        visualizations.run_visualizations()
    except pipeline.PipelineError:
        logger.error("Pipeline cannot complete; exiting", exc_info=True)
        return 1
    except Exception:  # last-resort guard so we exit gracefully, never with a raw traceback
        logger.error("Unexpected failure; exiting", exc_info=True)
        return 1
    logger.info("All stages completed successfully")
    return 0


if __name__ == "__main__":
    # Re-import under the name "main" so log lines show a clear module name
    from main import main as _main

    sys.exit(_main())
