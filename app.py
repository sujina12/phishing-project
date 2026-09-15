from flask import Flask, jsonify, request
from predictor import PhishingPredictor

app = Flask(__name__)

# Load once when the server starts.
predictor = PhishingPredictor()


# @app.get("/")
# def home():
#     return jsonify({"message": "Phishing API is running"})
@app.get("/")
def home():
    return "Phishing API is running"


@app.post("/api/analyse")
def analyse():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Provide non-empty email text"}), 400

    try:
        result = predictor.predict_with_explanation(text)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)