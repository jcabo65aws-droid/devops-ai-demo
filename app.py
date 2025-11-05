from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Decide backend IA según disponibilidad
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

@app.route("/")
def home():
    return jsonify({
        "message": "🚀 DevOps AI Demo running",
        "model": "OpenAI" if OPENAI_API_KEY else "Transformers local"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]

    try:
        if OPENAI_API_KEY:
            # Prompt sencillo
            prompt = f"Classify the sentiment of this text as Positive, Neutral or Negative: {text}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            output = response.choices[0].message["content"].strip()
        else:
            result = sentiment_pipeline(text)[0]
            output = f"{result['label']} (score={result['score']:.2f})"

        return jsonify({
            "input": text,
            "prediction": output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 80))
    app.run(host="0.0.0.0", port=port)
