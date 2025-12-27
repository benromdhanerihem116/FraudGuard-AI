from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# 1. LOAD THE SAVED BRAINS 
print(" Loading FraudGuard AI System...")

try:
    scaler = joblib.load('scaler.pkl')
    autoencoder = load_model('autoencoder_model.h5')
    encoder = load_model('encoder_model.h5')
    classifier = load_model('hybrid_model.h5')
    print(" System Online: All models loaded successfully.")
except Exception as e:
    print(f" ERROR: Could not load models.\nDetail: {e}")

# 2. DATA PREPARATION FUNCTION
def prepare_input(amount, time, v_features):
    # Scale Time and Amount (The scaler expects shape (n, 2))
    to_scale = np.array([[time, amount]])
    scaled_values = scaler.transform(to_scale)[0] # Returns [Scaled_Time, Scaled_Amount]
    
    scaled_time = scaled_values[0]
    scaled_amount = scaled_values[1]
    
    # Reassemble the 30 features in the correct order
    final_vector = np.array([scaled_time] + v_features + [scaled_amount])
    
    # Reshape to (1, 30) because Keras expects a batch
    return final_vector.reshape(1, -1)

# 3. THE PREDICTION ENDPOINT 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Get inputs
        amount = data.get('amount')
        time = data.get('time')
        v_features = data.get('features') 
        
        # SECURITY ADDED HERE 
        # 1. Verify that fields exist (are not None)
        if amount is None or time is None:
            return jsonify({
                "error": "Missing data: 'amount' and 'time' fields are required."
            }), 400

        # 2. Verify that inputs are numbers (int or float)
        # This prevents crashing if someone sends "amount": "lot of money"
        if not isinstance(amount, (int, float)) or not isinstance(time, (int, float)):
            return jsonify({
                "error": "Invalid format: 'amount' and 'time' must be numbers."
            }), 400
        
        # Validation: Ensure we have 28 V-features
        if not v_features or len(v_features) != 28:
            return jsonify({"error": "Invalid input: 'features' must contain exactly 28 values (V1-V28)."}), 400

        # A. PREPROCESS 
        X_input = prepare_input(amount, time, v_features)
        
        # B. PIPELINE STEP 1: AUTOENCODER (Get MSE)
        reconstructions = autoencoder.predict(X_input, verbose=0)
        # Calculate Reconstruction Error (MSE)
        mse_error = np.mean(np.power(X_input - reconstructions, 2), axis=1)
        
        # C. PIPELINE STEP 2: ENCODER (Get Latent Features) 
        latent_features = encoder.predict(X_input, verbose=0)

        # D. PIPELINE STEP 3: COMBINE (Hybrid) 
        # Stack Latent Features + MSE Error
        # Shape becomes: [Latent_1, Latent_2..., MSE]
        X_hybrid = np.hstack((latent_features, mse_error.reshape(-1, 1)))
        
        # E. PIPELINE STEP 4: CLASSIFY 
        fraud_probability = classifier.predict(X_hybrid, verbose=0)[0][0]
        
        # F. DECISION 
        # Using the threshold 0.7 
        THRESHOLD = 0.7
        is_fraud = float(fraud_probability) > THRESHOLD
        
        response = {
            "prediction": "FRAUD" if is_fraud else "LEGIT",
            "probability": float(fraud_probability),
            "anomaly_score": float(mse_error[0]),
            "status": "BLOCKED" if is_fraud else "APPROVED",
            "message": "High risk transaction detected" if is_fraud else "Transaction verified safe"
        }
        
        return jsonify(response)

    except Exception as e:
        # In production, log the error here for the developer
        print(f"Internal Error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)