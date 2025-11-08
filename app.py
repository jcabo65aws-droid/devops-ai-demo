from flask import Flask, request, jsonify, send_file
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Configuración
# ---------------------------
PORT = int(os.getenv("PORT", 8080))  # el contenedor escucha en 8080
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modelo por defecto: multilingüe con 3 clases (NEG/NEU/POS)
HF_MODEL = os.getenv("HF_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")

# ---------------------------
# Backends (carga perezosa)
# ---------------------------
sentiment_backend = "transformers"
sentiment_pipeline = None
openai_client = None


def get_transformers_pipeline():
    """Carga perezosa del pipeline de Transformers."""
    global sentiment_pipeline
    if sentiment_pipeline is None:
        from transformers import pipeline
        app.logger.info(f"Cargando modelo Transformers: {HF_MODEL}")
        sentiment_pipeline = pipeline("sentiment-analysis", model=HF_MODEL)
    return sentiment_pipeline


def normalize_label(label: str) -> str:
    """Normaliza etiquetas de distintos modelos a NEGATIVE/NEUTRAL/POSITIVE."""
    l = str(label).upper().strip()
    # Variantes comunes (HuggingFace suele usar LABEL_0/1/2)
    if l in ("POSITIVE", "POS", "LABEL_2"):
        return "POSITIVE"
    if l in ("NEGATIVE", "NEG", "LABEL_0"):
        return "NEGATIVE"
    if l in ("NEUTRAL", "NEU", "LABEL_1"):
        return "NEUTRAL"
    return l


# Backend OpenAI (opcional)
if USE_OPENAI and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        sentiment_backend = "openai"
        app.logger.info("Backend OpenAI activado.")
    except Exception as e:
        app.logger.warning(f"No se pudo inicializar OpenAI ({e}). Se usará Transformers.")
        USE_OPENAI = False
        sentiment_backend = "transformers"


# ---------------------------
# Rutas HTTP
# ---------------------------
@app.route("/", methods=["GET"])
def root():
    """Sirve la UI. Busca index.html en la raíz o en /static."""
    base = os.path.dirname(__file__)
    p1 = os.path.join(base, "index.html")               # raíz del proyecto
    p2 = os.path.join(base, "static", "index.html")     # fallback a /static
    if os.path.exists(p1):
        return send_file(p1)
    if os.path.exists(p2):
        return send_file(p2)
    return (
        "<h1>DevOps + AI Demo</h1><p>UI no encontrada.</p>",
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.route("/health", methods=["GET", "HEAD"])
def health():
    """Health check simple para balanceadores/monitores."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint de inferencia:
    - Entrada: JSON {"text": "..."}
    - Salida: {"backend": "...", "prediction": NEG/NEU/POS, "scores": {...}}
    """
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Campo 'text' es requerido"}), 400

        # --- Backend OpenAI (opcional) ---
        if sentiment_backend == "openai" and openai_client is not None:
            prompt = (
                "Clasifica el sentimiento del siguiente texto como "
                "NEGATIVE, NEUTRAL o POSITIVE y entrega scores 0..1 para cada clase.\n"
                "Responde sólo JSON con las llaves: prediction, positive, negative, neutral.\n\n"
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

            return jsonify(
                {
                    "backend": "openai",
                    "prediction": prediction,
                    "scores": {
                        "positive": round(pos, 6),
                        "negative": round(neg, 6),
                        "neutral": round(neu, 6),
                    },
                }
            ), 200

        # --- Backend Transformers (gratis) ---
        pipe = get_transformers_pipeline()
        # return_all_scores=True para obtener todas las clases (NEG/NEU/POS)
        outputs = pipe(text, truncation=True, return_all_scores=True)[0]  # lista de dicts

        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        best_label, best_score = "NEUTRAL", 0.0

        for item in outputs:
            lab = normalize_label(item.get("label", "NEUTRAL"))
            sc = float(item.get("score", 0.0))
            if lab == "POSITIVE":
                scores["positive"] = sc
            elif lab == "NEGATIVE":
                scores["negative"] = sc
            elif lab == "NEUTRAL":
                scores["neutral"] = sc
            if sc > best_score:
                best_label, best_score = lab, sc

        return jsonify(
            {
                "backend": "transformers",
                "prediction": best_label,
                "scores": {
                    "positive": round(scores["positive"], 6),
                    "negative": round(scores["negative"], 6),
                    "neutral": round(scores["neutral"], 6),
                },
            }
        ), 200

    except Exception as e:
        app.logger.exception("Error en /predict")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Para ejecución local (en contenedor usamos gunicorn)
    app.run(host="0.0.0.0", port=PORT)
