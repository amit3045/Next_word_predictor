# Configuration file for backend
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

# Default model currently used for prediction.
# BERT can be added later in MODEL_MAP without breaking the app.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "new_fine")

MODEL_MAP = {
    "new_fine": {
        "model": os.path.join(MODELS_DIR, "new_fine.h5"),
        "tokenizer": os.path.join(MODELS_DIR, "new_fine.pkl"),
    },
    "new_lstm": {
        "model": os.path.join(MODELS_DIR, "new_lstm.h5"),
        "tokenizer": os.path.join(MODELS_DIR, "new_lstm.pkl"),
    },
    # Future BERT model can be added here when ready:
    # "bert": {"model": os.path.join(MODELS_DIR, "bert_model.h5"), "tokenizer": os.path.join(MODELS_DIR, "bert_tokenizer.pkl")}
}

MODEL_CHOICES = list(MODEL_MAP.keys())

# Flask server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
