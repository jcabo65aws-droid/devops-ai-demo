from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
model = pipeline("sentiment-analysis")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")
    result = model(text)
    return jsonify(result)

@app.route("/")
def home():
    return "🤖 IA Service Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
