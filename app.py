from flask import Flask, request, jsonify
import joblib
import os

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
    # Bind host to 0.0.0.0 and use PORT env variable provided by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
