from flask import Flask, request, jsonify, send_from_directory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

app = Flask(__name__, static_folder="static", static_url_path="")
analyzer = SentimentIntensityAnalyzer()

@app.route("/")
def root():
    # Sirve index.html estático
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    scores = analyzer.polarity_scores(text)
    comp = scores["compound"]
    if comp >= 0.05:
        label = "Positive"
    elif comp <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return jsonify({
        "input": text,
        "prediction": label,
        "scores": scores
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

