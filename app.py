from flask import Flask, request, jsonify, send_file
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Config
# ---------------------------
PORT = int(os.getenv("PORT", 8080))  # contenedor escucha en 8080; host publica 80->8080
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modelo por defecto multilingüe con 3 clases; puedes cambiar por env HF_MODEL
HF_MODEL = os.getenv("HF_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")

# ---------------------------
# Backends (lazy)
# ---------------------------
sentiment_backend = "transformers"
sentiment_pipeline = None
openai_client = None

def get_transformers_pipeline():
    global sentiment_pipeline
    if sentiment_pipeline is None:
        from transformers import pipeline
        app.logger.info(f"Cargando modelo Transformers: {HF_MODEL}")
        sentiment_pipeline = pipeline("sentiment-analysis", model=HF_MODEL)
    return sentiment_pipeline

def normalize_label(label: str) -> str:
    l = str(label).upper().strip()
    # Mapear variantes comunes
    if l in ("POSITIVE", "POS", "LABEL_2"):
        return "POSITIVE"
    if l in ("NEGATIVE", "NEG", "LABEL_0"):
        return "NEGATIVE"
    if l in ("NEUTRAL", "NEU", "LABEL_1"):
        return "NEUTRAL"
    # fallback
    return l

# (OpenAI opcional)
if USE_OPENAI and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        sentiment_backend = "openai"
        app.logger.info("Backend OpenAI activado.")
    except Exception as e:
        app.logger.warning(f"No se pudo inicializar OpenAI ({e}). Uso Transformers.")
        USE_OPENAI = False
        sentiment_backend = "transformers"

# ---------------------------
# Rutas
# ---------------------------
@app.route("/", methods=["GET"])
def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return ("<h1>DevOps + AI Demo</h1><p>UI no encontrada.</p>", 200,
            {"Content-Type": "text/html; charset=utf-8"})

@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Campo 'text' es requerido"}), 400

        if sentiment_backend == "openai" and openai_client is not None:
            prompt = (
                "Clasifica el sentimiento del siguiente texto como NEGATIVE, NEUTRAL o POSITIVE "
                "y da scores 0..1 para cada uno. Responde JSON con prediction, positive, negative, neutral.\n"
                f"Texto: {text}\n"
            )
            completion = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = completion.choices[0].message.content
            import json
            try:
                parsed = json.loads(content)
                prediction = normalize_label(parsed.get("prediction", "NEUTRAL"))
                pos = float(parsed.get("positive", 0.33))
                neg = float(parsed.get("negative", 0.33))
                neu = float(parsed.get("neutral", 0.34))
            except Exception:
                prediction, pos, neg, neu = "NEUTRAL", 0.34, 0.33, 0.33

            return jsonify({
                "backend": "openai",
                "prediction": prediction,
                "scores": {"positive": pos, "negative": neg, "neutral": neu},
            }), 200

        # Transformers (gratis) - lazy init + return_all_scores
        pipe = get_transformers_pipeline()
        out = pipe(text, truncation=True, return_all_scores=True)[0]  # lista de dicts
        scores = {"positive": None, "negative": None, "neutral": None}
        best_label, best_score = "NEUTRAL", 0.0

        for item in out:
            lab = normalize_label(item.get("label", "NEUTRAL"))
            sc = float(item.get("score", 0.0))
            if lab == "POSITIVE":
                scores["positive"] = sc
            elif lab == "NEGATIVE":
                scores["negative"] = sc
            elif lab == "NEUTRAL":
                scores["neutral"] = sc
            # track best
            if sc > best_score:
                best_label, best_score = lab, sc

        # Rellenar faltantes (si el modelo es binario)
        for k in ("positive", "negative", "neutral"):
            if scores[k] is None:
                scores[k] = 0.0

        return jsonify({
            "backend": "transformers",
            "prediction": best_label,
            "scores": {
                "positive": round(scores["positive"], 6),
                "negative": round(scores["negative"], 6),
                "neutral": round(scores["neutral"], 6),
            },
        }), 200

    except Exception as e:
        app.logger.exception("Error en /predict")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
