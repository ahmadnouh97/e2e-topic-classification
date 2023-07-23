import mlflow
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()

DATA_DIR = Path(BASE_DIR, "data")
RAW_DATA_DIR = Path(DATA_DIR, "raw")
PROCESSED_DATA_DIR = Path(DATA_DIR, "processed")
ASSETS_DIR = Path(BASE_DIR, "assets")
CONFIG_DIR = Path(BASE_DIR, "config")

EMOJI_DICT_FILE = str(Path(ASSETS_DIR, "Emoji_Dict.p"))
PROCESSED_DATA_FILE = str(Path(PROCESSED_DATA_DIR, "tweet_topic_processed.csv"))
PREDICTIONS_DATA_FILE = str(Path(PROCESSED_DATA_DIR, "predictions.csv"))
TIPS_FILE = str(Path(ASSETS_DIR, "tips.json"))

STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")


with open(str(Path(ASSETS_DIR, "stopwords.txt")), "r") as file:
    STOPWORDS = [line.strip() for line in file.readlines()]

MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))