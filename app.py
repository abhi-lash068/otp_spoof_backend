from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)  # ✅ allow Flutter app to call this backend

# Load your model
model = joblib.load('otp_spoof_model.pkl')

# Simple API key (replace with your own secret key)
API_KEY = os.getenv("API_KEY", "mysecretkey123")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ API Key check
        auth = request.headers.get("Authorization")
        if not auth or auth != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(force=True)
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" in request'}), 400

        message = data['message']
        prediction = model.predict([message])[0]
        return jsonify({'label': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
