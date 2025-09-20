# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('otp_spoof_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
