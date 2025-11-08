from flask import Flask, request, jsonify, send_file
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Config
# ---------------------------
PORT = int(os.getenv("PORT", 80))
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Model backends
# ---------------------------
sentiment_backend = "transformers"
sentiment_pipeline = None
openai_client = None

if USE_OPENAI and OPENAI_API_KEY:
    try:
        # OpenAI opcional (requiere OPENAI_API_KEY y USE_OPENAI=1)
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        sentiment_backend = "openai"
        app.logger.info("Using OpenAI backend for sentiment.")
    except Exception as e:
        app.logger.warning(f"No se pudo inicializar OpenAI ({e}). Uso Transformers.")
        USE_OPENAI = False

if not USE_OPENAI:
    # Transformers (gratis). Modelo por defecto pequeño y popular.
    from transformers import pipeline
    HF_MODEL = os.getenv("HF_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_pipeline = pipeline("sentiment-analysis", model=HF_MODEL)
    app.logger.info(f"Using Transformers backend: {HF_MODEL}")

# ---------------------------
# Rutas
# ---------------------------
@app.route("/", methods=["GET"])
def root():
    """
    Sirve index.html si existe, si no devuelve una página simple.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return (
        "<h1>DevOps + AI Demo</h1>"
        "<p>Desplegado automáticamente con CI/CD sobre AWS.</p>",
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )

@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Body JSON: { "text": "me encanta este laboratorio" }
    Respuesta: {
      "backend": "transformers" | "openai",
      "prediction": "POSITIVE" | "NEGATIVE",
      "scores": { "positive": float, "negative": float }
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Campo 'text' es requerido"}), 400

        if sentiment_backend == "openai" and openai_client is not None:
            # Clasificación simple vía OpenAI (si se activó explícitamente)
            # Prompt muy directo para binario positivo/negativo
            prompt = (
                "Clasifica el sentimiento del siguiente texto como POSITIVE o NEGATIVE y da un score 0..1.\n"
                f"Texto: {text}\n"
                "Responde en JSON con las llaves: prediction, positive, negative."
            )
            completion = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = completion.choices[0].message.content
            # Intento robusto: si no es JSON perfecto, mapeamos heurísticamente
            import json
            try:
                parsed = json.loads(content)
                prediction = str(parsed.get("prediction", "POSITIVE")).upper()
                pos = float(parsed.get("positive", 0.5))
                neg = float(parsed.get("negative", 0.5))
            except Exception:
                # fallback muy simple
                prediction = "POSITIVE" if "POS" in content.upper() else "NEGATIVE"
                pos = 0.9 if prediction == "POSITIVE" else 0.1
                neg = 1.0 - pos

            return jsonify({
                "backend": "openai",
                "prediction": prediction,
                "scores": {"positive": pos, "negative": neg},
            }), 200

        # Default: Transformers (gratis)
        result = sentiment_pipeline(text, truncation=True)[0]
        label = str(result.get("label", "POSITIVE")).upper()
        score = float(result.get("score", 0.5))
        if label.startswith("NEG"):
            pos = 1.0 - score
            neg = score
            prediction = "NEGATIVE"
        else:
            pos = score
            neg = 1.0 - score
            prediction = "POSITIVE"

        return jsonify({
            "backend": "transformers",
            "prediction": prediction,
            "scores": {"positive": round(pos, 6), "negative": round(neg, 6)},
        }), 200

    except Exception as e:
        app.logger.exception("Error en /predict")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Escucha en 0.0.0.0:80 para que el contenedor publique en el puerto 80 del host
    app.run(host="0.0.0.0", port=PORT)
