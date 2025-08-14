import os
import sys
import time
import json
import logging
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_and_preprocess
from scorer import make_pred, get_top_features, plot_density

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info('Service initialized')

    def _save_all_outputs(self, submission, pred_proba, base_name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_ts   = os.path.join(self.output_dir, f"predictions_{ts}_{base_name}.csv")
        json_ts  = os.path.join(self.output_dir, f"top_features_{ts}_{base_name}.json")
        plot_ts  = os.path.join(self.output_dir, f"density_plot_{ts}_{base_name}.png")

        submission.to_csv(csv_ts, index=False)
        with open(json_ts, "w", encoding="utf-8") as f:
            json.dump(get_top_features(5), f, ensure_ascii=False, indent=2)
        plot_density(pred_proba, plot_ts)

        logger.info("Saved timestamped: %s, %s, %s", csv_ts, json_ts, plot_ts)

        csv_fixed  = os.path.join(self.output_dir, "sample_submission.csv")
        json_fixed = os.path.join(self.output_dir, "top_features.json")
        plot_fixed = os.path.join(self.output_dir, "density_plot.png")

        submission.to_csv(csv_fixed, index=False)
        with open(json_fixed, "w", encoding="utf-8") as f:
            json.dump(get_top_features(5), f, ensure_ascii=False, indent=2)
        plot_density(pred_proba, plot_fixed)

        logger.info("Saved fixed: %s, %s, %s", csv_fixed, json_fixed, plot_fixed)

    def process_single_file(self, file_path: str):
        try:
            logger.info('Processing file: %s', file_path)
            X, original_index = load_and_preprocess(file_path)
            submission, pred_proba = make_pred(X, original_index)
            base_name = os.path.basename(file_path).split('.')[0]
            self._save_all_outputs(submission, pred_proba, base_name)
            logger.info('Done for: %s', file_path)
        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.info('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()

    for fname in os.listdir(service.input_dir):
        if fname.endswith(".csv"):
            service.process_single_file(os.path.join(service.input_dir, fname))

    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
