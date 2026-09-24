from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import os
import tempfile
import h5py
from functools import lru_cache
from config import DEFAULT_MODEL, MODEL_MAP, HOST, PORT

app = Flask(__name__)


def load_legacy_model(model_path):
    try:
        return load_model(model_path, compile=False)
    except ValueError as error:
        if "batch_shape" not in str(error):
            raise

        with h5py.File(model_path, "r") as source:
            model_config = json.loads(source.attrs["model_config"])
            for layer in model_config["config"]["layers"]:
                if layer["class_name"] == "InputLayer" and "batch_shape" in layer["config"]:
                    batch_shape = layer["config"].pop("batch_shape")
                    layer["config"]["input_shape"] = batch_shape[1:]

            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as converted:
                converted_path = converted.name

            with h5py.File(converted_path, "w") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                target.attrs["model_config"] = json.dumps(model_config).encode("utf-8")
                source.copy("model_weights", target)

        try:
            return load_model(converted_path, compile=False)
        finally:
            os.unlink(converted_path)

@lru_cache(maxsize=None)
def get_model_and_tokenizer(model_name):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(MODEL_MAP.keys())}")

    model_path = MODEL_MAP[model_name]["model"]
    tokenizer_path = MODEL_MAP[model_name]["tokenizer"]

    model = load_legacy_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    seed_text = data.get('text', '')
    model_name = data.get('model', DEFAULT_MODEL)

    if not seed_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        model, tokenizer = get_model_and_tokenizer(model_name)

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        if not token_list:
            return jsonify({"error": "No valid words found in the text for this model."}), 400

        token_list = pad_sequences([token_list], maxlen=3, padding='pre')
        preds = model.predict(token_list)[0]

        top_indices = preds.argsort()[-3:][::-1]
        index_word = {v: k for k, v in tokenizer.word_index.items()}
        top_words = [index_word.get(i, '') for i in top_indices]
        top_words = [word for word in top_words if word]

        return jsonify(top_words)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False)
